import torch
import os
import loralib as loralb
from .convert_to_compvis import convert_unet_state_dict, convert_text_enc_state_dict

try:
    from safetensors.torch import save_file, safe_open
except: 
    print("Safetensors is not installed. Saving while using use_safetensors will fail.")

UNET_REPLACE = ["Transformer2DModel", "ResnetBlock2D"]
TEXT_ENCODER_REPLACE = ["CLIPEncoderLayer"]

UNET_ATTENTION_REPLACE = ["CrossAttention"]
TEXT_ENCODER_ATTENTION_REPLACE = ["CLIPAttention"]

"""
Copied from: https://github.com/cloneofsimo/lora/blob/bdd51b04c49fa90a88919a19850ec3b4cf3c5ecd/lora_diffusion/lora.py#L189
"""
def find_modules(
        model,
        ancestor_class= None,
        search_class = [torch.nn.Linear],
        exclude_children_of = [loralb.Linear, loralb.Conv2d],
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

def create_lora_linear(child_module, r, dropout=0):
    return loralb.Linear(
        child_module.in_features, 
        child_module.out_features, 
        r=r,
        lora_dropout=dropout
    )

def create_lora_conv(child_module, r):
    return loralb.Conv2d(
                child_module.in_channels, 
                child_module.out_channels,
                kernel_size=child_module.kernel_size[0],
                padding=child_module.padding,
                r=r,
            )

def add_lora_to(
    model, 
    target_module=UNET_REPLACE, 
    is_text=False,
    search_class=[torch.nn.Linear], 
    r=32, 
    lora_bias='none'
):
    for module, name, child_module in find_modules(
        model, 
        ancestor_class=target_module, 
        search_class=search_class
    ):
        # Drop 10% of the text conditioning to improve classifier free guidance
        dropout = 0.1 if is_text else 0

        # Check if the child module of the model is type Linear or Conv2d.
        if isinstance(child_module, torch.nn.Linear):
            l = create_lora_linear(child_module, r, dropout)

        if isinstance(child_module, torch.nn.Conv2d):
            l = create_lora_conv(child_module, r)

        # Check if child module of the model has bias.
        if child_module.bias is not None: 
            l.bias = child_module.bias
        
        # Assign the frozen weight of model's Linear or Conv2d to the LoRA model.
        l.weight =  module._modules[name].weight

        # Replace the new LoRA model with the model's Linear or Conv2d module.
        module._modules[name] = l

    # Unfreeze only the newly added LoRA weights, but keep the model frozen.
    loralb.mark_only_lora_as_trainable(model, bias=lora_bias)

def save_lora(
        unet=None, 
        text_encoder=None, 
        save_text_weights=False,
        output_dir="output",
        lora_filename="lora.safetensors",
        use_safetensors=True, 
        lora_bias='none', 
        save_for_webui=True
    ):

        # Create directory for the full LoRA weights.
        trainable_weights_dir = f"{output_dir}/full_weights"
        lora_out_file_full_weight = f"{trainable_weights_dir}/{lora_filename}"
        os.makedirs(trainable_weights_dir, exist_ok=True)

        # Create LoRA out filename.
        lora_out_file = f"{output_dir}/{lora_filename}"

        # Get save method depending on use_safetensors
        save_method = save_file if use_safetensors else torch.save
        
        if use_safetensors:
            ext = '.safetensors'
            save_path_full_weights = lora_out_file_full_weight.replace('.pt', ext)
            save_path = lora_out_file.replace('.pt',ext)
        else:
            ext = '.pt'
            save_path_full_weights = lora_out_file_full_weight.replace('.safetensors', ext)
            save_path = lora_out_file.replace('.safetensors', ext)

        for i, model in enumerate([unet, text_encoder]):
            if save_text_weights and i == 1:
                save_path_full_weights = save_path_full_weights.replace(ext, f"_text{ext}")
                
            # Load only the LoRAs from the state dict.
            lora_dict = loralb.lora_state_dict(model, bias=lora_bias)
            
            # Save the models as fp32. This ensures we can finetune again without having to upcast.                      
            save_method(lora_dict, save_path_full_weights)

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

            save_method(lora_dict_fp16, save_path.replace(ext, f"_webui{ext}"))

# The non webui weights should be called here. Only the full weights will work
# Load after instantiating LoRA weights to the model. 
def load_lora(model, lora_path: str):
    try:
        if os.path.exists(lora_path):

            if lora_path.endswith('.safetensors'):
                lora_dict = safe_open(lora_path, framework='pt')
            else:
                lora_dict = torch.load(lora_path)

            model.load_state_dict(lora_dict, strict=False)

    except Exception as e:
        print(f"Could not load your lora file: {e}")

def set_mode(model, train=False):
    is_train = False
    is_eval = False
    for n, m in model.named_modules():
        is_lora = isinstance(m, loralb.Linear) or isinstance(m, loralb.Conv2d)
        if is_lora:
            if train:
                is_train = True
            else:
                is_eval = True
                m.train(train)
    
    if is_train: print("Train mode enabled for LoRA.")
    if is_eval: print("Evaluation mode enabled for LoRA.")
