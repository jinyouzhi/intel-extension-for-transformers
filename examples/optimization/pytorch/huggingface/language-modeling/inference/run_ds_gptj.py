
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex
import time
import os
import sys
import argparse
import deepspeed


# args
parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
parser.add_argument('--precision', default='bf16', type=str, help="fp32 or bf16")
parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")
parser.add_argument('--greedy', action='store_true')
args = parser.parse_args()
print(args)

amp_enabled = True if args.precision != "fp32" else False
amp_dtype = torch.bfloat16 if args.precision != "fp32" else torch.float32

if args.greedy:
    generate_kwargs = dict(do_sample=False, temperature=0.9)
else:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# load model
model_id_online = "EleutherAI/gpt-j-6B"
model_id_offline_model = "./gpt-j-6B-model"
model_id_offline_token = "./gpt-j-6B-tokenizer"
if os.path.exists(model_id_offline_model):
    model = AutoModelForCausalLM.from_pretrained(model_id_offline_model, low_cpu_mem_usage=True)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id_online, low_cpu_mem_usage=True)
    model.save_pretrained(model_id_offline_model)

if os.path.exists(model_id_offline_token):
    tokenizer = AutoTokenizer.from_pretrained(model_id_offline_token)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id_online_token)
    tokenizer.save_pretrained(model_id_offline_token)

model = model.eval()

# to channels last
model = model.to(memory_format=torch.channels_last)
#model = ipex.optimize(model, dtype=amp_dtype, inplace=True)

engine = deepspeed.init_inference(model=model, mp_size=1, dtype=torch.bfloat16, replace_with_kernel_inject=False)
model = engine.module

# input prompt
# prompt = "Once upon a time,"
# 32 tokens input
prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
         " She wanted to go to places and meet new people, and have fun."

# start
total_time = 0.0
num_iter = 10
num_warmup = 3
#for i in range(num_iter):
#    tic = time.time()
#    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#    gen_tokens = model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
#    gen_text = tokenizer.batch_decode(gen_tokens)[0]
#    toc = time.time()
#    print(gen_text, flush=True)
#    if i >= num_warmup:
#        total_time += (toc - tic)
with torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        toc = time.time()
        print(gen_text, flush=True)
        if i >= num_warmup:
            total_time += (toc - tic)

print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))

