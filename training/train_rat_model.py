import os
import sys

# os.environ['HF_HOME'] = '/mnt/nas/sarvam/open_llm/huggingface_models/'
os.environ['HF_HOME'] = "/raid/infolab/sarvam/elk_nas_sarvam/open_llm/huggingface_models/"
sys.path.insert(1, '/root/CS726/')

import re
import json
from constants import *
from transformers.generation.utils import *
from utils.utils import *
from utils import ra_postproc
from anytree import Node, RenderTree
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from peft import PeftModel, LoraModel, LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import random

# Generate 100 random numbers between 0 and 5500
random_numbers = [random.randint(0, 5500) for _ in range(200)]

device = torch.device("cuda:3") if torch.cuda.is_available() else "cpu"

peft_model_id = "codes/3b_full_schema_bert"

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "transform_layer"],
    lora_dropout=0.01,
)


model, tokenizer = load_model(model_type="codes_3b")

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# model = PeftModel.from_pretrained(model, peft_model_id) # Load the model with PEFT

# embedding model
subtree_embed_model, subtree_embed_tokenizer = load_embed_model()
subtree_embed_model.eval()

# Move the models to the device
subtree_embed_model.to(device)
model.to(device)


# Add special tokens
def add_special_tokens():

    tokenizer.add_special_tokens({'additional_special_tokens': [NODE_BEGIN_TOKEN, NODE_END_TOKEN, LCHILD_NODE_TOKEN, RCHILD_NODE_TOKEN]})
    tokenizer.add_special_tokens({'additional_special_tokens': [f"st{i}" for i in range(0, NUM_ST_NODES)]})

    model.resize_token_embeddings(len(tokenizer))


# Create embeddings for the subtree traversal orders from CLS token
def create_embedding(traversal_order):
    encoded_input = subtree_embed_tokenizer(traversal_order, return_tensors='pt').to(device)

    cls_embedding = None

    with torch.no_grad():
        outputs = subtree_embed_model(**encoded_input, output_hidden_states=True)
        # Extract the last hidden state of the token `[CLS]` for classification tasks
        cls_embedding = outputs.hidden_states[-1][0][0]

    return cls_embedding


# Implementation
add_special_tokens()

with open("/root/CS726/data/spider/rat_input_data/train_schema_subsetting_full_schema.json", "r") as f:
    data = json.load(f)

filtered_data = [item for item in data if item.get('id') in random_numbers]

print(f"Number of items in the filtered data: {len(filtered_data)}")

optimizer = AdamW(lora_model.parameters(), lr=5e-5)


epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for item in tqdm(filtered_data):

        join_count = (item['SQL'].lower().split()).count('join')
        if join_count > 2:
            continue

        level_node_embeddings = []
        lora_model.count_of_subtrees_in_previous_level = len(item['input_subtree_traversal_list'])

        try:       
            for k in item['input_subtree_traversal_list']:

                post_order_embed = create_embedding("postorder:" + list(k.values())[0]['post'])
                pre_order_embed = create_embedding("preorder:" + list(k.values())[0]['pre'])
                in_order_embed = create_embedding("inorder:" + list(k.values())[0]['in'])
                level_node_embeddings.append(torch.concat((post_order_embed, pre_order_embed, in_order_embed)))

            if len(item['input_subtree_traversal_list']) != 0:
                lora_model.update_lm_heads(level_node_embeddings)
        
        except:
            continue

        

        inputs = tokenizer(item['model_input'] + tokenizer.eos_token + item['output'], return_tensors='pt')
        optimizer.zero_grad()
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = input_ids.clone()

        outputs = lora_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss:.2f}')


    
lora_model.save_pretrained("codes_3b/full_schema_bert_sample_150")