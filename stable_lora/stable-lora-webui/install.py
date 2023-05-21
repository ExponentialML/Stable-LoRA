import launch
import os
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
launch.run_pip(f"install -r {req_file}", "Installing Stable LoRA Requirements")
