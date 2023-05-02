# Stable LoRA (WIP)
Train LoRA using Microsoft's official implementation on Diffusion Models.

This is the most efficient and easiest way to train LoRAs without the added complexity while also being shareable between libraries and implementations. 

## Getting Started

Clone this repository into your training folder:

```bash
git clone https://github.com/ExponentialML/Stable-LoRA
```

Install the LoRA from the official repository:

```bash
pip install git+https://github.com/microsoft/LoRA
```

Simply add it to your model before the training loop.

```python
# Import function to add the LoRA, as well as target modules.
from stable_lora.lora import add_lora_to, UNET_REPLACE, TEXT_ENCODER_REPLACE

from diffusers import StableDiffusionPipeline

# Load a Stable Diffusion Model
pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)

# Add LoRA to the UNET
add_lora_to(unet, target_module=UNET_REPLACE, search_class=[torch.nn.Linear, torch.nn.Conv2d], r=32)

# Add LoRA to the Text Encoder
add_lora_to(text_encoder, target_module=TEXT_ENCODER_REPLACE, r=32)

# Your optimizers and training code...
```

After adding the LoRA to your model, you can easily add parameter wise optimizer params if needed.

```python
# Example

unet_params = []
text_encoder_params = []

for n, p in unet.named_parameters():
    if 'lora' in n:
        unet_params.append({
            "params": p, 
            "lr": args.learning_rate
        })

for n, p in text_encoder.named_parameters():
    if 'lora' in n:
        text_encoder_params.append({
            "params": p, 
            "lr": args.learning_rate_text
        })

optimizer_params = unet_params + text_encoder_params
```

Saving can be done using safetensors or the traditional way, using `.pt` files:

```python
from stable_lora.lora import add_lora_to, UNET_REPLACE, TEXT_ENCODER_REPLACE

save_lora(unet, path='save_file_path.safetensors')
save_lora(text_encoder=text_encoder, use_safetensors=False, path='save_file_path.pt')

```

## Tips

This has been tested on Stable Diffusion 1.5 Based models.

It is reccomended to try `5e-6`, `3e-5`, and `1e-4` learning rates. 

The default rank is `32`, but can be set by passing it through the `r` parameter. Using lower ranks will consume less memory, with high ranks consuming more. 
This also contributes to file save sizes.

These factors will be solely dependent on your task, but are a good starting point.

## TODO
- [ ] Add Diffusers Training Scripts for Dreambooth and Finetuning.
- [x] Implement saving and Loading LoRA's (`PT` & `safetensors`).
- [ ] Add as a Stable Diffusion webui Extension for inference.
- [ ] Possible integrations for webui training.

## Credits
[cloneofsimo](https://github.com/cloneofsimo/lora) For their LoRA implementation code.
[Microsoft](https://github.com/microsoft/LoRA) For the official code.
