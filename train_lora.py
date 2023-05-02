# Bootstrapped from:
# https://github.com/cloneofsimo/lora/blob/master/training_scripts/train_lora_dreambooth.py

import argparse
import hashlib
import itertools
import math
import os
import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import json

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    Transformer2DModel
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from stable_lora.lora import add_lora_to, save_lora, set_mode, \
    UNET_REPLACE, TEXT_ENCODER_REPLACE

from pathlib import Path

import random
import re

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        color_jitter=False,
        h_flip=False,
        resize=False,
        dataset_norm=False
    ):
        self.size = size
        self.center_crop = center_crop
        self.color_jitter = color_jitter
        self.h_flip = h_flip
        self.tokenizer = tokenizer
        self.resize = resize
        self.dataset_norm = dataset_norm

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        
        self.image_transforms = self.compose()
        self.normalized_mean_std = self.get_dataset_norm(class_data_root)

    def get_dataset_norm(self, class_data_root):
        if self.dataset_norm:
            imgs_to_process = self.instance_images_path
            
            if class_data_root is not None:
                imgs_to_process += self.class_images_path

            mean = 0
            std = 0
            
            for img in tqdm(imgs_to_process, desc="Processing image normalization..."):
                img = Image.open(img).convert("RGB")
                img = self.image_transforms(img)
                
                mean += img.mean()
                std += img.std()

            mean = [mean / len(imgs_to_process)]
            std = [std / len(imgs_to_process)]

            print(f"Dataset mean and std are: {mean}, {std}")
            
            return mean, std
        else:
            return [0.5], [0.5]

    def compose(self):
        img_transforms = []

        if self.resize:
            img_transforms.append(
                transforms.Resize(
                    (self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if self.center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if self.color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if self.h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        return transforms.Compose([*img_transforms, transforms.ToTensor()])
        
    def image_transform(self, img): 
        img_composed = self.image_transforms(img)

        if not self.dataset_norm:
            mean = 0.5 if img_composed.mean() > 0.5 else img_composed.mean()
            std = 0.5 if img_composed.std() > 0.5 else img_composed.std()

            mean, std = [mean], [std]
        else:
            mean, std = self.normalized_mean_std

        return transforms.Normalize(mean, std)(img_composed)

    def open_img(self, index, folder): 
        img = Image.open(folder[index % self.num_instance_images])

        if not img.mode == "RGB":
            img = img.convert("RGB")

        return img

    def tokenize_prompt(self, prompt):
        return self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

    def get_train_sample(self, index, example, base_name, folder, prompt):
        image = self.open_img(index, self.instance_images_path)
        example[f"{base_name}_images"] = self.image_transform(image)
        example[f"{base_name}_prompt_ids"] = self.tokenize_prompt(prompt)
        example[f"{base_name}_prompt"] = prompt

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        self.get_train_sample(
            index, 
            example, 
            "instance", 
            self.instance_images_path, 
            self.instance_prompt
        )

        if self.class_data_root:
            self.get_train_sample(
                index, 
                example, 
                "class", 
                self.class_images_path, 
                self.class_prompt
            )
        return example

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


logger = get_logger(__name__)

def prior_preservation_loss(model_pred, target, prior_loss_weight, mode="additive"):
    model_pred, prior_pred = torch.chunk(model_pred, 2, dim=0)
    model_target, prior_target = torch.chunk(target, 2, dim=0)

    loss = F.mse_loss(model_pred.float(), model_target.float(), reduction="mean")
    prior_loss = F.mse_loss(prior_pred.float(), prior_target.float(), reduction="mean")

    if mode is not None and mode == "additive":
        loss = loss + (prior_loss * prior_loss_weight)
    else:
        loss = loss + prior_loss_weight * prior_loss
    return loss

def set_mode_wrapper(unet=None, text_encoder=None, is_train=False):
    for model in [unet, text_encoder]:
        if model is not None:
            model.train() if is_train else model.eval()
            set_mode(model, train=is_train)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="",
        required=True,
        help="A JSON file with the same args as argparse (instance_data_dir, class_data_dir, etc.)",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--preview_prompt",
        type=str,
        default=None,
        help="The prompt to use when generating preview images",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--prior_preservation_mode",
        type=str,
        choices=["additive", "multiply"],
        default="additive",
        help=("The prior preservation loss mode."
            "Additive: loss + (prior_loss * loss_weight)"
            "Multiply: loss + loss_weight * prior_loss"
        )
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=800,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_for_webui",
        action="store_true",
        default=True,
        help="Save a LoRA model for usage in the AUTOMATIC1111 webui.",
    )
    parser.add_argument(
        "--preview_steps",
        type=int,
        default=100,
        help="Save preview every X updates steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="Whether or not to use bias when training LoRA.",
        choices=["none", "lora_only", "all"]
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataset_norm",
        action="store_true",
        default=False,
        help="Normalizes the entire dataset by calculating the mean and standard deviation of all elements.",
    )
    parser.add_argument(
        "--save_preview",
        action="store_true",
        default=False,
        help="Save preview images during training.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank. Not to be confused with LoRA rank.",
    )
    parser.add_argument(
        "--resume_unet",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    parser.add_argument(
        "--resume_text_encoder",
        type=str,
        default=None,
        help=("File path for text encoder lora to resume training."),
    )
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        required=False,
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args

def main(args):
    
    logging_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = (
                torch.float16 if accelerator.device.type == "cuda" else torch.float32
            )
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=args.sample_batch_size
            )

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_dir
                        / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    )
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    for model in [unet, vae, text_encoder]:
        model.requires_grad_(False)

    if args.use_xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.resume_unet or args.resume_text_encoder:
        logger.warn("Resuming training for LoRA is not yet implemented.")

    add_lora_to(
        unet, 
        target_module=UNET_REPLACE, 
        search_class=[torch.nn.Linear, torch.nn.Conv2d], 
        r=args.lora_rank,
        lora_bias=args.lora_bias
    )

    if args.train_text_encoder:
        add_lora_to(
            text_encoder, 
            target_module=TEXT_ENCODER_REPLACE, 
            r=args.lora_rank,
            lora_bias=args.lora_bias
        )

    if args.scale_lr:

        # Since prior preservation is batched, we do this to compensate for it.
        scale_batch_size = (
            args.train_batch_size if not args.with_prior_preservation
            else args.train_batch_size * 2
        )
        
        if (scale_batch_size % 2) != 0 and args.with_prior_preservation:
            raise ValueError("Batch size must be an even number when using prior preservation.")
            
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * scale_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    text_lr = (
        args.learning_rate
        if args.learning_rate_text is None
        else args.learning_rate_text
    )

    params_to_optimize = list(unet.parameters())

    if args.train_text_encoder: 
        params_to_optimize += list(text_encoder.parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    if os.path.exists(args.json_path):
        train_datasets = []
        json_file = json.load(open(args.json_path))

        if 'preview_prompt' in json_file:
            args.preview_prompt = json_file['preview_prompt']

        for train_data in json_file['data']:
            root = json_file['train_data_root']

            instance_data_dir = os.path.join(root,train_data['instance_data_dir'])
            class_dir = os.path.join(root,train_data['class_data_dir'])
            class_data_root = (
                class_dir if args.with_prior_preservation else None
                )
            train_datasets.append(
                DreamBoothDataset(
                    instance_data_root=instance_data_dir,
                    instance_prompt=train_data['instance_prompt'],
                    class_data_root=class_data_root,
                    class_prompt=train_data['class_prompt'],
                    tokenizer=tokenizer,
                    size=args.resolution,
                    center_crop=args.center_crop,
                    color_jitter=args.color_jitter,
                    resize=args.resize,
                    dataset_norm=args.dataset_norm
            ))
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    else:
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
            color_jitter=args.color_jitter,
            resize=args.resize,
            dataset_norm=args.dataset_norm
        )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        prompt = [example["instance_prompt"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "prompt": prompt
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    print(f"  Saving for webui = {args.save_for_webui}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    for epoch in range(args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader):
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()

            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            if args.with_prior_preservation:
                loss = prior_preservation_loss(model_pred, target, args.prior_loss_weight, mode=args.prior_preservation_mode)
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters())
                    if args.train_text_encoder
                    else unet.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            optimizer.zero_grad()

            global_step += 1
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                
                # Sample a preview of current training progress.
                if global_step % args.preview_steps == 0 and args.save_preview:
                    pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=unet,
                            text_encoder=text_encoder,
                            revision=args.revision,
                        )

                    pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                        args.pretrained_model_name_or_path, 
                        subfolder='scheduler'
                    )
                    pipeline = pipeline.to("cuda")  

                    with torch.no_grad():
                        if args.save_preview :
                            preview_dir = f"{args.output_dir}/previews"
                            os.makedirs(preview_dir, exist_ok=True)
                           
                            set_mode_wrapper(unet, text_encoder, is_train=False)
                             
                            prompt = batch['prompt'] if len(args.preview_prompt) <= 0 else args.preview_prompt 
                            image = pipeline(prompt, num_inference_steps=20, width=args.resolution, height=args.resolution).images[0]
                            image.save(f"{preview_dir}/{prompt}_{args.save_steps}_{last_save}_{global_step}.png")  

                            set_mode_wrapper(unet, text_encoder, is_train=True)
                    
                    del pipeline

                if args.save_steps and global_step - last_save >= args.save_steps:
                    if accelerator.is_main_process:
                  
                        lora_out_file = (
                            f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.safetensors"
                        )
                        save_lora(
                            unet, 
                            path=lora_out_file, 
                            lora_bias=args.lora_bias, 
                            save_for_webui=args.save_for_webui
                        )

                        if args.train_text_encoder:
                            lora_out_file_text = (
                                f"{args.output_dir}/text_lora_weight_e{epoch}_s{global_step}.safetensors"
                            )
                            save_lora(
                                text_encoder=text_encoder, 
                                path=lora_out_file_text, 
                                lora_bias=args.lora_bias, 
                                save_for_webui=args.save_for_webui
                            )
 
                        print(f"LoRA checkpoints have been saved to ${args.output_dir}")
                        last_save = global_step

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        print("Training complete.")
        accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
