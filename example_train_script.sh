# TRAINING DIRECTORIES
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export PROJECT_NAME="lora_training"
export TRAINING_ROOT="path/to/ALL_FOLDERS_OF_TRAINING_IMAGES/"
export OUTPUT_DIR="output/$PROJECT_NAME"

#path/to/ALL_FOLDERS_OF_TRAINING_IMAGES/folder_of_images
export INSTANCE_DIR="$TRAINING_ROOT/folder_of_images"
export CLASS_DIR="$TRAINING_ROOT/folder_of_class_images"

# TRAINING PROMPTS
export TOKEN="shld"
export PROMPT="a $TOKEN dog"
export CLASS_PROMPT="a dog"
export PREVIEW_PROMPT="a $TOKEN dog walking in New York City, high quality"

# TRAINING SETTINGS
export LORA_RANK=64
export NUM_CLASS_IMAGES=800
export TRAIN_BATCH=1
export SAVE_STEPS=500
export PREVIEW_STEPS=100
export MAX_TRAIN_STEPS=3000
export UNET_LR=1e-4
export ADAMW_WEIGHT_DECAY=1e-2
export PRIOR_LOSS_WEIGHT=1

# LAUNCH TRAINING (train_lora -h for all available params)
accelerate launch train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --lora_rank=$LORA_RANK \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=$NUM_CLASS_IMAGES \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="$PROMPT" \
  --prior_loss_weight=$PRIOR_LOSS_WEIGHT \
  --class_prompt="$CLASS_PROMPT" \
  --resolution=512 \
  --mixed_precision="fp16" \
  --train_text_encoder \
  --with_prior_preservation \
  --train_batch_size=$TRAIN_BATCH \
  --adam_weight_decay=$ADAMW_WEIGHT_DECAY \
  --learning_rate=$UNET_LR \
  --resize=True \
  --save_steps=$SAVE_STEPS \
  --preview_steps=$PREVIEW_STEPS \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=50 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --save_preview \
  --preview_prompt="$PREVIEW_PROMPT"
