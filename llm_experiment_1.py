#Task 1: Test the math abilities of LLM (1B vs. 8B)
#Multiplication (or other operations) between two numbers 
#Generate data

#Task 2: Compare the speed when using flash-attention (8B)

#Task 3: Compare the speed when enabling KV-cache (8B)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from random import randrange

model_name = '/mnt/pan/courses/llm24/xxh584/Meta-Llama-3.1-8B-Instruct'
# model_name = '/mnt/pan/courses/llm24/xxh584/Llama-3.2-1B'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16).cuda()
model.eval()

random_numbers = []
for _ in range(6):
    random_numbers.append(randrange(35, 90))

tracker=1
for j in range(len(random_numbers)):
    for k in range(len(random_numbers)):
        input_text = f'{random_numbers[j]} * {random_numbers[k]}='  

        input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()

        tokens = model.generate(input_ids, max_new_tokens=15, use_cache=True)
        output_text = tokenizer.decode(tokens[0], skip_special_tokens=True)

        print('ANSWER NUMBER ', str(tracker))
        print(output_text)
        tracker+=1
