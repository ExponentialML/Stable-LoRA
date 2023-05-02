# Stable LoRA (WIP)
Train LoRA using Microsoft's official implementation with Stable Diffusion models.
This is the most efficient and easiest way to train LoRAs without the added complexity while also being shareable between libraries and implementations. 

🚧 *This repo is still in active development. Expect bugs until the *WIP* flag is removed.* 🚧

## Getting Started

Clone this repository into your training folder:

```bash
git clone https://github.com/ExponentialML/Stable-LoRA
```

Install requirements from this repository:

```bash
pip install -r requirements.txt
```

Install the LoRA from the official repository:

```bash
pip install git+https://github.com/microsoft/LoRA
```

### Example

Training is done by using the Diffusers library. 

See [example_train_script.sh](https://github.com/ExponentialML/Stable-LoRA/blob/main/example_train_script.sh) for an example script of Dreambooth.

Running it is as easy as doing:

```python
accelerate launch train_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="instance_images" \
  --lora_rank=64 \
  --output_dir="./output" \
  --instance_prompt="a shld dog" \
  --resolution=512 \
  --mixed_precision="fp16" \
  --train_text_encoder \
  --learning_rate=1e-4 \
  --resize=True \
  --save_steps=200 \
  --preview_steps=100 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=50 \
  --max_train_steps=5000 \
  --save_preview \
  --preview_prompt="a shld dog as a super hero"
```

After training, results will be saved in your output folder. 

By default, a AUTOMATIC1111 webui variant is saved alongside them (webui extension is in development).

## Developers

Simply add it to your model before the training loop.

```python
# Import function to add the LoRA, as well as target modules.
from stable_lora.lora import add_lora_to, UNET_REPLACE, TEXT_ENCODER_REPLACE

from diffusers import StableDiffusionPipeline

# Load a Stable Diffusion Model
pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)

# Freeze the models. Remember, only the LoRA weights get trained, not the model itself.
pipeline.unet.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)

# Add LoRA to the UNET
add_lora_to(
    pipeline.unet, 
    target_module=UNET_REPLACE, 
    search_class=[torch.nn.Linear, torch.nn.Conv2d], 
    r=32
)

# Add LoRA to the Text Encoder
add_lora_to(pipeline.text_encoder, target_module=TEXT_ENCODER_REPLACE, r=32)

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
from stable_lora.lora import save_lora

save_lora(unet, path='save_file_path.safetensors')
save_lora(text_encoder=text_encoder, use_safetensors=False, path='save_file_path.pt')

```

Additionally, you can set the mode to evaluation or train mode.

```python
from stable_lora.lora import set_mode

# Train mode
set_mode(unet, is_train=True)

# Evaluation mode
set_mode(unet, is_train=False)
```

## Tips

This has been tested on Stable Diffusion 1.5 Based models.

It is recommended to try `5e-6`, `3e-5`, and `1e-4` learning rates. 

The default rank is `32`, but can be set by passing it through the `r` parameter. Using lower ranks will consume less memory, with high ranks consuming more. 
This also contributes to file save sizes.

These factors will be solely dependent on your task, but are a good starting point.

You can also enable biases to train as well. 

Depending on your data and learning rate, you may be able to squeeze extra performance out of training.
```python
# Example | Applicable to Unet or Text Encoder

lora_bias = 'lora_only'
add_lora_to(pipeline.text_encoder, target_module=TEXT_ENCODER_REPLACE, r=32, lora_bias=lora_bias)

# Must be set here as well when saving.
save_lora(text_encoder=text_encoder, use_safetensors=True, path='save_file_path.pt', lora_bias=lora_bias)
```

## TODO
- [ ] Add Diffusers Training Scripts for Dreambooth and Finetuning.
- [x] Implement saving and Loading LoRA's (`PT` & `safetensors`).
- [ ] Add as a Stable Diffusion webui Extension for inference.
- [ ] Possible integrations for webui training.

## Credits
[cloneofsimo](https://github.com/cloneofsimo/lora) For their LoRA implementation code.

[Microsoft](https://github.com/microsoft/LoRA) For the official code.
