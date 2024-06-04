# Based on a script from: https://github.com/huggingface/trl/issues/1303
# Run this with naive pipeline parallel PP with "python test_scripts/test_pp.py"
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from accelerate import PartialState
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


device_map = config.get('BASICS', 'device_map')

if device_map == "DDP":
    device_string = PartialState().process_index
    device_map={'':device_string}

# Load the dataset
dataset_name = config.get('BASICS', 'dataset')
dataset = load_dataset(dataset_name, split="train")

# Load the model + tokenizer
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = dataset_name = config.get('BASICS', 'model')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    cache_dir='',
    use_cache = False,
    device_map = device_map,
)

# PEFT config
lora_alpha = int(config.get('LORA', 'alpha'))
lora_dropout = float(config.get('LORA', 'dropout'))
lora_r = int(config.get('LORA', 'r'))
peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r = lora_r,
    bias = config.get('LORA', 'bias'),
    task_type = config.get('LORA', 'task'),
    target_modules = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save = ["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)

# Args 
max_seq_length = int(config.get('TRAINING', 'max_seq_length'))
output_dir = config.get('TRAINING', 'output_dir')
per_device_train_batch_size = int(config.get('TRAINING', 'per_device_train_batch_size'))
gradient_accumulation_steps = int(config.get('TRAINING', 'gradient_accumulation_steps'))
optim = config.get('TRAINING', 'optim')
save_steps = int(config.get('TRAINING', 'save_steps'))
logging_steps = int(config.get('TRAINING', 'logging_steps'))
learning_rate = float(config.get('TRAINING', 'learning_rate'))
max_grad_norm = float(config.get('TRAINING', 'max_grad_norm'))
max_steps = int(config.get('TRAINING', 'max_steps'))
warmup_ratio = float(config.get('TRAINING', 'warmup_ratio'))
lr_scheduler_type = config.get('TRAINING', 'lr_scheduler_type')
training_arguments = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    optim = optim,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = learning_rate,
    bf16 = bool(config.get('TRAINING', 'bf16')),
    max_grad_norm = max_grad_norm,
    max_steps = max_steps,
    warmup_ratio = warmup_ratio,
    group_by_length = True,
    lr_scheduler_type = lr_scheduler_type,
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": bool(config.get('TRAINING', 'reentrant'))}, #must be false for DDP
    report_to = "wandb",
)

# Trainer 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train
def train_model():
    trainer.train()
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    trainer.train()