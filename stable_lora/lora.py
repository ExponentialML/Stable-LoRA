import torch
import os
import loralib as loralb
from safetensors.torch import save_file

UNET_REPLACE = ["Transformer2DModel", "ResnetBlock2D"]
TEXT_ENCODER_REPLACE = ["CLIPEncoderLayer"]

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

def create_lora_linear(child_module, r):
    return loralb.Linear(
        child_module.in_features, 
        child_module.out_features, r=r, 
        merge_weights=True
    )

def create_lora_conv(child_module, r):
    return loralb.Conv2d(
                child_module.in_channels, 
                child_module.out_channels,
                kernel_size=child_module.kernel_size[0],
                padding=child_module.padding ,
                r=r, 
                merge_weights=True
            )

def add_lora_to(model, target_module=UNET_REPLACE, search_class=[torch.nn.Linear], r=32):
    for module, name, child_module in find_modules(
        model, 
        ancestor_class=target_module, 
        search_class=search_class
    ):
        # Check if the child module of the model is type Linear or Conv2d.
        if isinstance(child_module, torch.nn.Linear):
            l = create_lora_linear(child_module, r)

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
    loralb.mark_only_lora_as_trainable(model)

def get_lora_modules(model):
    lora_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k}
    return lora_dict

def save_lora(unet=None, text_encoder=None, use_safetensor=True, path='model.pt'):
    for model in [unet, text_encoder]:
        if model is not None:
            lora_dict = get_lora_modules(model)
            if use_safetensor:
                save_file(lora_dict, path.replace('.pt', '.safetensors'))
            else:
                torch.save(lora_dict, path.replace('.safetensors', '.pt'))

    
