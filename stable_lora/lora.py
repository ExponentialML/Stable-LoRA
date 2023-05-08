import torch
import os
import loralib as loralb
import json

from torch.utils.data import ConcatDataset
from transformers import CLIPTokenizer
from .convert_to_compvis import convert_unet_state_dict, convert_text_enc_state_dict

try:
    from safetensors.torch import save_file, safe_open
except: 
    print("Safetensors is not installed. Saving while using use_safetensors will fail.")

UNET_REPLACE = ["Transformer2DModel", "ResnetBlock2D"]
TEXT_ENCODER_REPLACE = ["CLIPAttention", "CLIPTextEmbeddings"]

UNET_ATTENTION_REPLACE = ["CrossAttention"]
TEXT_ENCODER_ATTENTION_REPLACE = ["CLIPAttention", "CLIPTextEmbeddings"]

"""
Copied from: https://github.com/cloneofsimo/lora/blob/bdd51b04c49fa90a88919a19850ec3b4cf3c5ecd/lora_diffusion/lora.py#L189
"""
def find_modules(
        model,
        ancestor_class= None,
        search_class = [torch.nn.Linear],
        exclude_children_of = [loralb.Linear, loralb.Conv2d, loralb.Embedding],
    ):
        """
        Find all modules of a certain class (or union of classes) that are direct or
        indirect descendants of other modules of a certain class (or union of classes).

        Returns all matching modules, along with the parent of those moduless and the
        names they are referenced by.
        """

        # Get the targets we should replace all linears under
        if ancestor_class is not None:
            ancestors = (
                module
                for module in model.modules()
                if module.__class__.__name__ in ancestor_class
            )
        else:
            # this, incase you want to naively iterate over all modules.
            ancestors = [module for module in model.modules()]

        # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
        for ancestor in ancestors:
            for fullname, module in ancestor.named_modules():
                if any([isinstance(module, _class) for _class in search_class]):
                    # Find the direct parent if this is a descendant, not a child, of target
                    *path, name = fullname.split(".")
                    parent = ancestor
                    while path:
                        parent = parent.get_submodule(path.pop(0))
                    # Skip this linear if it's a child of a LoraInjectedLinear
                    if exclude_children_of and any(
                        [isinstance(parent, _class) for _class in exclude_children_of]
                    ):
                        continue
                    # Otherwise, yield it
                    yield parent, name, module

def process_lora_metadata_dict(dataset):
    keys_to_exclude = [
        "center_crop", 
        "color_jitter", 
        "h_flip", 
        "instance_data_root",
        "instance_images_path",
        "class_images_path",
        "class_data_root",
        "dataset_norm",
        "image_transforms",
        "_length",
        "resize",
        "normalized_mean_std"
    ]
    return {
        k: str(v) for k, v in dataset.items() \
            if (k not in keys_to_exclude and \
                not isinstance(v, CLIPTokenizer))
        }

def create_lora_metadata(lora_name, train_dataset, r):
    import uuid
    is_concat_dataset = isinstance(train_dataset, ConcatDataset)
    dataset = (
        train_dataset.__dict__ if not is_concat_dataset
            else 
        [d.__dict__ for d in train_dataset.datasets]
    )   
    if is_concat_dataset:
        dataset = [process_lora_metadata_dict(x) for x in dataset]
    else:
        dataset = process_lora_metadata_dict(dataset)
    
    metadata = {
        "stable_lora": "v1", 
        "lora_name": lora_name + "_" + uuid.uuid4().hex.lower()[:5],
        "train_dataset": json.dumps(dataset, indent=4)
    }
    return metadata

def create_lora_linear(child_module, r, dropout=0, bias=False, scale=0):
    return loralb.Linear(
        child_module.in_features, 
        child_module.out_features, 
        merge_weights=False,
        bias=bias,
        lora_dropout=dropout,
        lora_alpha=r,
        r=r
    )
    return lora_linear

def create_lora_conv(child_module, r, dropout=0, bias=False, rescale=False, scale=0):
    return loralb.Conv2d(
        child_module.in_channels, 
        child_module.out_channels,
        kernel_size=child_module.kernel_size[0],
        padding=child_module.padding,
        merge_weights=False,
        bias=bias,
        lora_dropout=dropout,
        lora_alpha=r,
        r=r,
    )
    return lora_conv    

def create_lora_emb(child_module, r):
    return loralb.Embedding(
        child_module.num_embeddings, 
        child_module.embedding_dim, 
        merge_weights=False,
        lora_alpha=r,
        r=r
    )

def activate_lora_train(model, bias):
    def unfreeze():
        print(model.__class__.__name__ + " LoRA set for training.")
        return loralb.mark_only_lora_as_trainable(model, bias=bias)

    return unfreeze

def add_lora_to(
    model, 
    target_module=UNET_REPLACE, 
    is_text=False,
    search_class=[torch.nn.Linear], 
    r=32, 
    dropout=0,
    lora_bias='none'
):
    for module, name, child_module in find_modules(
        model, 
        ancestor_class=target_module, 
        search_class=search_class
    ):
        bias = hasattr(child_module, "bias")

        # Check if the child module of the model is type Linear or Conv2d.
        if isinstance(child_module, torch.nn.Linear):
            l = create_lora_linear(child_module, r, dropout, bias=bias)

        if isinstance(child_module, torch.nn.Conv2d):
            l = create_lora_conv(child_module, r, dropout, bias=bias)

        if isinstance(child_module, torch.nn.Embedding):
            l = create_lora_emb(child_module, r)
            
        # Check if child module of the model has bias.
        if bias:
            l.bias = child_module.bias
        
        # Assign the frozen weight of model's Linear or Conv2d to the LoRA model.
        l.weight =  child_module.weight

        # Replace the new LoRA model with the model's Linear or Conv2d module.
        module._modules[name] = l

    # Unfreeze only the newly added LoRA weights, but keep the model frozen.
    return activate_lora_train(model, lora_bias)

def save_lora(
        unet=None, 
        text_encoder=None, 
        save_text_weights=False,
        output_dir="output",
        lora_filename="lora.safetensors",
        lora_bias='none', 
        save_for_webui=True,
        only_webui=False,
        metadata=None
    ):
        if not only_webui:
            # Create directory for the full LoRA weights.
            trainable_weights_dir = f"{output_dir}/full_weights"
            lora_out_file_full_weight = f"{trainable_weights_dir}/{lora_filename}"
            os.makedirs(trainable_weights_dir, exist_ok=True)

        ext = '.safetensors'
        # Create LoRA out filename.
        lora_out_file = f"{output_dir}/{lora_filename}{ext}"

        if not only_webui:
            save_path_full_weights = lora_out_file_full_weight

        save_path = lora_out_file

        if not only_webui:
            for i, model in enumerate([unet, text_encoder]):
                if save_text_weights and i == 1:
                    save_path_full_weights = save_path_full_weights.replace(ext, f"_text{ext}")
                    
                # Load only the LoRAs from the state dict.
                lora_dict = loralb.lora_state_dict(model, bias=lora_bias)
                
                # Save the models as fp32. This ensures we can finetune again without having to upcast.                      
                save_file(lora_dict, save_path_full_weights)

        if save_for_webui:
            
            # Convert the keys to compvis model and webui
            unet_lora_dict = loralb.lora_state_dict(unet, bias=lora_bias) 
            lora_dict_fp16 = convert_unet_state_dict(unet_lora_dict)
            
            if save_text_weights:
                text_encoder_dict = loralb.lora_state_dict(text_encoder, bias=lora_bias)
                lora_dict_text_fp16 = convert_text_enc_state_dict(text_encoder_dict)
                
                # Update the Unet dict to include text keys.
                lora_dict_fp16.update(lora_dict_text_fp16)

            # Cast tensors to fp16. It's assumed we won't be finetuning these.
            for k, v in lora_dict_fp16.items():
                lora_dict_fp16[k] = v.to(dtype=torch.float16)

            save_file(
                lora_dict_fp16, 
                save_path, 
                metadata=metadata
            )

def load_lora(model, lora_path: str):
    try:
        if os.path.exists(lora_path):
            lora_dict = safe_open(lora_path, framework='pt')
            model.load_state_dict(lora_dict, strict=False)

    except Exception as e:
        print(f"Could not load your lora file: {e}")

def set_mode(model, train=False):
    for n, m in model.named_modules():
        is_lora = any(
            isinstance(m,x) for x in [loralb.Linear, loralb.Conv2d, loralb.Embedding]
        )
        if is_lora:
            m.train(train)

def set_mode_group(models, train):
   for model in models: 
        set_mode(model)
        model.train(train)