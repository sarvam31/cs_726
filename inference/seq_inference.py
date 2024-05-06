import os
import sys

# os.environ['HF_HOME'] = '/mnt/nas/sarvam/open_llm/huggingface_models/'
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
device = torch.device("cuda:3") if torch.cuda.is_available() else "cpu"


model, tokenizer = load_model(model_type="codes_3b", location = '/root/hf_models/codes_3b_full_schema_bert_sample_200_model')

# embedding model
subtree_embed_model, subtree_embed_tokenizer = load_embed_model()
subtree_embed_model.eval()

# Move the models to the device
subtree_embed_model.to(device)
model.to(device)


# Add special tokens


# Create embeddings for the subtree traversal orders from CLS token
def create_embedding(traversal_order):
    encoded_input = subtree_embed_tokenizer(traversal_order, return_tensors='pt').to(device)

    cls_embedding = None

    with torch.no_grad():
        outputs = subtree_embed_model(**encoded_input, output_hidden_states=True)
        # Extract the last hidden state of the token `[CLS]` for classification tasks
        cls_embedding = outputs.hidden_states[-1][0][0]

    return cls_embedding




with open("/root/CS726/data/spider/rat_input_data/dev_schema_subsetting_full_schema.json", "r") as f:
    data = json.load(f)





for item in tqdm(data):

    join_count = (item['SQL'].lower().split()).count('join')
    if join_count > 2:
        continue

    level_node_embeddings = []
    model.count_of_subtrees_in_previous_level = len(item['input_subtree_traversal_list'])

    try:       
        for k in item['input_subtree_traversal_list']:

            post_order_embed = create_embedding("postorder:" + list(k.values())[0]['post'])
            pre_order_embed = create_embedding("preorder:" + list(k.values())[0]['pre'])
            in_order_embed = create_embedding("inorder:" + list(k.values())[0]['in'])
            level_node_embeddings.append(torch.concat((post_order_embed, pre_order_embed, in_order_embed)))

        if len(item['input_subtree_traversal_list']) != 0:
            model.update_lm_heads(level_node_embeddings)
    
    except:
        continue

    

    inputs = tokenizer(item['model_input'], return_tensors='pt', padding=True).to(device)



    outputs = model.generate(inputs['input_ids'], pad_token_id=tokenizer.eos_token_id, max_new_tokens=50)
    text = [tokenizer.batch_decode(i, skip_special_tokens=True) for i in [outputs]]

    responses = ["".join(i) for i in text]

    print(responses)




    
