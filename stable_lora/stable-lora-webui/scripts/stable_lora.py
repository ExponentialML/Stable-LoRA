import loralib as loralb
import torch
import glob

from safetensors.torch import load_file
import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks
from modules.shared import opts, cmd_opts, state, cmd_opts, sd_model
from modules.sd_models import read_state_dict

EXTENSION_NAME = "Stable LoRA"

def lora_files():
    paths = \
        glob.glob(os.path.join(cmd_opts.lora_dir, '**/*.pt'), recursive=True) + \
        glob.glob(os.path.join(cmd_opts.lora_dir, '**/*.safetensors'), recursive=True)

    paths = sorted(paths)
    paths.insert(0, "None")
    return paths
    
class StableLoraScript(scripts.Script):

    def __init__(self):
        self.lora_loaded = 'lora_loaded' 
        self.previous_lora_alpha = 0
        self.current_sd_checkpoint = ""
        
    def lora_linear_forward(self, weight, lora_A, lora_B, alpha, *args):
        return (lora_B @ lora_A) * alpha / min(lora_A.shape)

    def lora_conv_forward(self, weight, lora_A, lora_B, alpha, *args):
        return (lora_B @ lora_A).view(weight.shape) * alpha / min(lora_A.shape)

    def is_lora_loaded(self, sd_model):
        return hasattr(sd_model, self.lora_loaded)

    def handle_lora_load(self, sd_model, set_lora_loaded=False):
        if not hasattr(sd_model, self.lora_loaded) and set_lora_loaded:
            setattr(sd_model, self.lora_loaded, True)

        if hasattr(sd_model, self.lora_loaded) and not set_lora_loaded:
            delattr(sd_model, self.lora_loaded)

    def title(self):
            return EXTENSION_NAME

    def show(self, is_img2img):
            return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        lora_files_list = lora_files()
        with gr.Accordion(EXTENSION_NAME, open=False):
                with gr.Row():
                    lora_dir = gr.Dropdown(
                        label="LoRA 1",
                        choices=lora_files_list,
                        value=lora_files_list[0],
                    )
                with gr.Row():
                    lora_dir_2 = gr.Dropdown(
                        label="LoRA 2",
                        choices=lora_files_list,
                        value=lora_files_list[0],
                    )
                with gr.Row():
                    lora_alpha = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=1,
                            step=0.1,
                            label="LoRA Alpha"
                    )
                    
        return [lora_alpha, lora_dir, lora_dir_2]

    @torch.no_grad()
    def process(self, p, lora_alpha, lora_file_1, lora_file_2):

        # Load multiple LoRAs
        lora_files = [lora_file_1, lora_file_2]
        lora_files_list = []

        # Save the previous alpha value so we can re-run the LoRA with new values.        
        alpha_changed = (lora_alpha != self.previous_lora_alpha) \
            and self.is_lora_loaded(p.sd_model)

        # If the LoRA is still loaded, unload it.
        if all(lf == "None" for lf in lora_files) and self.is_lora_loaded(p.sd_model) \
            or p.sd_model.sd_checkpoint_info.filename != self.current_sd_checkpoint:

            model_dict = read_state_dict(p.sd_model.sd_checkpoint_info.filename)
            p.sd_model.load_state_dict(model_dict,  strict=False)

            self.handle_lora_load(p.sd_model, set_lora_loaded=False)
            print(f"Unloaded previously loaded LoRA files")

            self.current_sd_checkpoint = p.sd_model.sd_checkpoint_info.filename
            return

        can_activate_lora = not self.is_lora_loaded(p.sd_model) or \
            p.sd_model.sd_checkpoint_info.filename != self.current_sd_checkpoint
        
        # Process LoRA
        if can_activate_lora or alpha_changed:
            
            self.previous_lora_alpha = lora_alpha            
            lora_alpha = lora_alpha

            lora_files = [lf for lf in lora_files if lf != "None"]
            if len(lora_files) == 0: return

            for lora_file in lora_files:
                LORA_FILE = lora_file.split('/')[-1]
                LORA_DIR = cmd_opts.lora_dir
                LORA_PATH = f"{LORA_DIR}/{LORA_FILE}"

                lora_model_text_path = f"{LORA_DIR}/text_{LORA_FILE}"
                lora_text_exists = os.path.exists(lora_model_text_path)
                
                is_safetensors = LORA_PATH.endswith('.safetensors')
                load_method = load_file if is_safetensors else torch.load
                
                lora_model = load_method(LORA_PATH)

                lora_files_list.append(lora_model)

            
            model_dict = read_state_dict(p.sd_model.sd_checkpoint_info.filename)
            p.sd_model.load_state_dict(model_dict,  strict=False)

            for n, m in p.sd_model.named_modules():
                for lora_model in lora_files_list:
                    for k, v in lora_model.items():

                        # If there is bias in the LoRA, add it here.
                        if 'bias' in k and n == k.split('.bias')[0]:
                            
                            if m.bias is None:
                                m.bias = torch.nn.Parameter(v.to(m.weight.device, dtype=m.weight.dtype))
                            else:
                                m.bias.weight = v.to(m.weight.device, dtype=m.weight.dtype)

                        if 'lora_A' in k and n == k.split('.lora_A')[0]:
                            lora_A = lora_model[f"{n}.lora_A"].to(m.weight.device, dtype=m.weight.dtype)
                            lora_B = lora_model[f"{n}.lora_B"].to(m.weight.device, dtype=m.weight.dtype)
                    
                            if isinstance(m, torch.nn.Linear):
                                m.weight += self.lora_linear_forward(m.weight, lora_A, lora_B, lora_alpha)

                            if isinstance(m, torch.nn.Conv2d):
                                m.weight += self.lora_conv_forward(m.weight, lora_A, lora_B, lora_alpha)          

                            continue

            self.handle_lora_load(p.sd_model, set_lora_loaded=True)
            
            for lora_file in lora_files:
                if alpha_changed:
                    print(f"Alpha changed for {lora_file.split('/')[-1]}.")
                if self.is_lora_loaded(p.sd_model):
                    print(f"LoRA loaded from {lora_file.split('/')[-1]}")
            