from datasets import load_dataset
import os 
# os.system("pip install -r requirements.txt")



train_dataset=load_dataset('jsonl',data_files="s3://sagemaker-us-west-2-979105248912/ambuje/zero_shot_entities/mix_data/train.jsonl"))
#eval_dataset = load_dataset('csv', data_files= os.path.join(os.environ["SM_CHANNEL_TEST"],"cricket_data_E2E_clubbed_test_data_1.csv"))

import torch
hf_token="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"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#model_id="meta-llama/Llama-2-7b-chat-hf"
model_id="mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# # Load model and tokenizer
accelerator = Accelerator()
# # Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_id,add_eos_token=False,add_bos_token=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# def generate_and_tokenize_prompt(data_point):
#     return tokenizer(data_point['prompt'])
# tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
# tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=800,
        lora_dropout=0.05,init_lora_weights= True,
        r=400,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
)
# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(accelerator.device)
from transformers import TrainingArguments


args = TrainingArguments(
    output_dir="/opt/ml/model",
    num_train_epochs=7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    learning_rate=1e-5,
    fp16=False,
    tf32=False,
    bf16_full_eval=True,
    #fp16_full_eval=True,
    # max_grad_norm=0.3,
    # warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True, # disable tqdm since with packing values are in correct,
    save_total_limit=10  ,
    # evaluation_strategy="epoch", 
    evaluation_strategy="no",do_eval=False,#,per_device_eval_batch_size=2,eval_accumulation_steps=4,
    #eval_steps=10,
    logging_steps=10,#dataloader_num_workers=accelerator.num_processes
    #report_to="tensorboard"
)
from trl import SFTTrainer

# max sequence length for model and packing of the dataset
from trl import SFTTrainer

# max sequence length for model and packing of the dataset
max_seq_length = 2048 
#accelerator = Accelerator()
trainer = accelerator.prepare(SFTTrainer(
    model=model,
    train_dataset=train_dataset['train'],
    # test_dataset=tokenized_val_dataset['train'],
    #eval_dataset=eval_dataset['train'],
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    args=args,
    dataset_text_field='text'
))
# train
trainer.train() 
# save model
trainer.save_model()
# import os
# import json
# import logging
# import argparse
# from datasets import load_dataset

# os.system(
#     "git clone https://github.com/FlagOpen/FlagEmbedding.git"
# )
# os.chdir("FlagEmbedding")
# os.system("pip install  .")

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()

#     # hyperparameters sent by the client are passed as command-line arguments to the script.
#     parser.add_argument("--num_nodes", type=int, default=1)
#     parser.add_argument("--nproc_per_node", type=int, default=4)
#     parser.add_argument("--model_name_or_path",type=str,default='BAAI/bge-large-en-v1.5')
#     parser.add_argument("--lr", type=float, default=2e-05)
#     parser.add_argument("--num_epochs", type=int, default=2) 
#     parser.add_argument("--batch_size_training", type=int, default=6)
#     # Data, model, and output directories
#     parser.add_argument("--output_data_dir", type=str, default="/opt/ml/model")
#     #parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
#     #parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
#     parser.add_argument("--train_data", type=str, default="s3://sagemaker-us-west-2-979105248912/ambuje/dapr/data/pre_train_malomatia/train.jsonl")
#     #parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

#     args, _ = parser.parse_known_args()
#     #print(args.target_modules)
#     #print(type(args.target_modules))
#     # os.system(
#     #     f"torchrun --nnodes {args.num_nodes} --nproc_per_node {args.nproc_per_node} examples/finetuning.py --model_name {args.model_name} --enable_fsdp --use_peft --peft_method {args.peft_method} --dataset custom_dataset --custom_dataset.file examples/custom_dataset.py:get_uniphore_dataset --save_model --fsdp_config.pure_bf16 --output_dir {os.environ['SM_MODEL_DIR']} --use_fast_kernels --lora_config.r {args.lora_rank} --lora_config.lora_alpha {args.lora_alpha} --lora_config.lora_dropout {args.lora_dropout} --lora_config.target_modules \"{','.join(args.target_modules)}\" --lr {args.lr} --num_epochs {args.num_epochs} --batch_size_training {args.batch_size_training}"
#     # )
    
#     os.system(
#         f"torchrun --nproc_per_node {args.nproc_per_node} -m FlagEmbedding.baai_general_embedding.retromae_pretrain.run --output_dir {args.output_data_dir} --model_name_or_path {args.model_name_or_path}  --train_data {args.train_data} --learning_rate {args.lr} --num_train_epochs {args.num_epochs} --per_device_train_batch_size {args.batch_size_training} --dataloader_drop_last True --max_seq_length 512 --logging_steps 10 "
#     )
