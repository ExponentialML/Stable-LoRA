# Bootstrapped from:
# https://github.com/cloneofsimo/lora/blob/master/training_scripts/train_lora_dreambooth.py

import hashlib
import itertools
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import json
import random
import re

from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm
from PIL import Image

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
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from stable_lora.lora import add_lora_to, save_lora, create_lora_metadata, \
    UNET_REPLACE, TEXT_ENCODER_REPLACE, \
    UNET_ATTENTION_REPLACE, TEXT_ENCODER_ATTENTION_REPLACE

from stable_lora.utils.dataset import DreamBoothDataset, PromptDataset
from stable_lora.utils.train_args import parse_args

logger = get_logger(__name__)

def generate_class_images(args):
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

def prior_preservation_loss(model_pred, target, prior_loss_weight, mode="additive"):
    
    if mode in ["additive", "multiply"]:
        model_pred_chunk, prior_pred = torch.chunk(model_pred, 2, dim=0)
        model_target, prior_target = torch.chunk(target, 2, dim=0)

        loss = F.mse_loss(model_pred_chunk.float(), model_target.float(), reduction="mean")
        prior_loss = F.mse_loss(prior_pred.float(), prior_target.float(), reduction="mean")

    if mode == "additive":
        loss = loss + (prior_loss * prior_loss_weight)

    elif mode == "multiply":
        loss = loss + prior_loss_weight * prior_loss

    elif mode in ["single_pass", "text"]:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    else:
        raise ValueError("Incorrect mode for prior preservation loss")
    
    return loss

def token_tuning(word: str, class_token: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel):
    with torch.no_grad():
        token = tokenizer(f"{word}", add_special_tokens=False).input_ids[0]
        class_token = tokenizer(f"{class_token}", add_special_tokens=False).input_ids[0]

        class_token_weight = text_encoder.text_model.embeddings.token_embedding.weight[class_token, :]
        text_encoder.text_model.embeddings.token_embedding.weight[token, :] = class_token_weight

        print(f"{token} has been initialized with {class_token}")

def collate_fn(examples, tokenizer: CLIPTokenizer):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        prompt = [example["instance_prompt"] for example in examples]
        img_name = [example["instance_img_name"] for example in examples]

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
            "prompt": prompt,
            "instance_img_name": img_name
        }
        return batch

def main(args):
    
    logging_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(args.output_dir, exist_ok=True)

    generate_class_images(args)

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
        num_hidden_layers=args.clip_layers
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
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.resume_unet or args.resume_text_encoder:
        logger.warn("Resuming training for LoRA is not yet implemented.")

    unet_replace = UNET_ATTENTION_REPLACE if args.only_attn else UNET_REPLACE
    text_replace = TEXT_ENCODER_ATTENTION_REPLACE if args.only_attn else TEXT_ENCODER_REPLACE

    #token_tuning(args.instance_prompt, args.class_prompt, tokenizer, text_encoder)
    
    train_text_encoder_lora = None
    train_unet_lora = add_lora_to(
        unet, 
        target_module=unet_replace, 
        search_class=[torch.nn.Linear, torch.nn.Conv2d], 
        r=args.lora_rank,
        lora_bias=args.lora_bias
    )

    if args.train_text_encoder:
        train_text_encoder_lora = add_lora_to(
            text_encoder, 
            is_text=True,
            target_module=text_replace,
            search_class=[torch.nn.Linear, torch.nn.Embedding],
            r=args.lora_rank,
            dropout=args.dropout,
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, tokenizer),
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

    # Create LoRA metadata that contains train parameters
    lora_metadata = create_lora_metadata(
        args.lora_name,
        train_dataset=train_dataset,
        r=args.lora_rank
    )

    # Unfreeze the LoRA parameters for training.
    for train_lora in [train_unet_lora, train_text_encoder_lora]:
        if train_lora is not None:
            train_lora()

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
    lora_save_msg = "Pending"
    global_step = 0
    last_save = 0

    for epoch in range(args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader):
            #unet.train()
            #if args.train_text_encoder:
            #    text_encoder.train()

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

            # TODO Test prior preservation based on text encoding positioning
            
            # Get the text embedding for conditioning
            input_ids = batch["input_ids"]
            if args.prior_preservation_mode == "text" and args.with_prior_preservation:
                
                # Don't compute gradients for our class prompt
                with torch.no_grad():
                    class_hidden_states = text_encoder(input_ids[1].unsqueeze(0))[0]
                    class_hidden_states /=  class_hidden_states.norm(dim=-1, keepdim=True)

                # Compute them for our instance prompt
                encoder_hidden_states = text_encoder(input_ids[0].unsqueeze(0))[0]
            
                # Use class prompt as base starting point for instance prompt
                encoder_hidden_states = (
                    class_hidden_states + (encoder_hidden_states * args.prior_loss_weight)
                )
                
                # We only need to process the instance batch
                noisy_latents = noisy_latents[0].unsqueeze(0)
                noise = noise[0].unsqueeze(0)
                timesteps = timesteps[0].unsqueeze(0)

            else:
                encoder_hidden_states = text_encoder(input_ids)[0]

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
                loss = prior_preservation_loss(
                    model_pred, 
                    target, 
                    args.prior_loss_weight, 
                    mode=args.prior_preservation_mode
                )              
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
                           
                            prompt = batch['prompt'][0] if len(args.preview_prompt) <= 0 else args.preview_prompt 
                            image = pipeline(prompt, num_inference_steps=20, width=args.resolution, height=args.resolution).images[0]
                            image.save(f"{preview_dir}/{last_save}_g{global_step}_{prompt}.png")  

                    del pipeline

                if args.save_steps and global_step - last_save >= args.save_steps:
                    if accelerator.is_main_process:
                        lora_out_file = f"{args.lora_name}_e{epoch}_s{global_step}.safetensors"
                        save_lora(
                            unet=unet, 
                            text_encoder=text_encoder,
                            save_text_weights=args.train_text_encoder,
                            output_dir=args.output_dir,
                            lora_filename=lora_out_file,
                            lora_bias=args.lora_bias, 
                            save_for_webui=args.save_for_webui,
                            only_webui=args.only_webui,
                            metadata=lora_metadata
                        )
                        lora_save_msg = f"LoRA saved to {args.output_dir}."
                        last_save = global_step

            logs = {
                "Loss:": loss.detach().item(), 
                "LR": lr_scheduler.get_last_lr()[0],
                "Last Save": lora_save_msg
                }   
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