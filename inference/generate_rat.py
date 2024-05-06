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



model, tokenizer = load_model(model_type="codes_3b", location = '/root/hf_models/codes_3b_full_schema_bert_sample_200_model')

subtree_embed_model, subtree_embed_tokenizer = load_embed_model()



level_node_embeddings = []  # 2D list of embeddings of nodes at each level
kth_level_subtrees = [] # 2D list of anytree objects at each level

# def add_special_tokens():
#     # TODO: add special tokens
#     # TODO: add st0-st100

#     # tokenizer.add_special_tokens({'additional_special_tokens': [NODE_BEGIN_TOKEN, NODE_END_TOKEN, BREAK_NODE_TOKEN]}) TODO: in training
#     tokenizer.add_special_tokens({'additional_special_tokens': [f"st{i}" for i in range(0, 50)]})

#     model.resize_token_embeddings(len(tokenizer))


# def cross_encoder(question, schema):

#     # TODO: call cross encoder and get ranked schema
#     # if required process it and return it
#     responses = None

#     return "How many singers do we have? | singer : singer.singer_id , singer.name , singer.country , singer.age , singer.song_name , singer.song_release_year , singer.is_male | stadium : stadium.location , stadium.name , stadium.capacity , stadium.highest , stadium.lowest , stadium.average , stadium.stadium_id | concert : concert.theme , concert.year , concert.concert_id , concert.concert_name , concert.stadium_id | singer_in_concert : singer_in_concert.concert_id , singer_in_concert.singer_id | concert.stadium_id = stadium.stadium_id | singer_in_concert.singer_id = singer.singer_id | singer_in_concert.concert_id = concert.concert_id"


def post_process_rat_to_sql(rat):
    pass
    # TODO: post process RAT to SQL, check smbop post_process ra to sql method for reference
    # TODO: might have to change smbop method to make it work for our case

def generate_kth_level_nodes(input_with_schema, level):

    model.level = level

    if level == 0:
        # prepare input for generating leaf nodes with question + ranked schema (format: [node_begin]node_content[node_end])
        
        model_input = input_with_schema


    if level >= 1:

        count_of_subtrees_in_previous_level = len(level_node_embeddings[-1])

        model.count_of_subtrees_in_previous_level = count_of_subtrees_in_previous_level

        embeddings = model.get_input_embeddings() 


        # TODO: set previous level subtrees count for llm's lm heads
        for i in range(0, count_of_subtrees_in_previous_level):
            token_id = tokenizer.convert_tokens_to_ids(f"st{i}")
            embeddings.weight.data[token_id] = level_node_embeddings[-1][i]
            # TODO: set st1... embeddings to generate nodes at kth level


        # prepare input for generate nodes with question + schema + previous level nodes embeddings obtained from bert (format: [node_begin]operation[NB]previous level subtree reference([NB]previous level subtree reference)?[node_end])
        
        model_input = input_with_schema + "|" + "|".join([f"st{i}" for i in range(count_of_subtrees_in_previous_level)]) # level node embeddings: [k subtrees, dim of embeddings]

    # generate nodes at kth level
    inputs = tokenizer(model_input, return_tensors="pt", padding=True).to(device)

    outputs = model.generate(inputs['input_ids'], pad_token_id=tokenizer.eos_token_id, max_new_tokens=500)
    text = [tokenizer.batch_decode(i, skip_special_tokens=True) for i in outputs]


    responses = ["".join(i) for i in text]
    responses_wo_input = [x[(x.find(NODE_BEGIN_TOKEN)):] for x in responses]

    return responses_wo_input

def get_previous_level_node_objects(left_subtree=None, right_subtree=None, level=None):

    if level == 0:
        return None, None
    
    assert left_subtree is not None, f"Error in node {left_subtree}. Left subtree not found"

    left_subtree_obj = kth_level_subtrees[level-1][int(left_subtree.replace('st', ''))]

    if right_subtree is not None:
        right_subtree_obj = kth_level_subtrees[level-1][int(right_subtree.replace('st', ''))]

    return left_subtree_obj, right_subtree_obj

def validate_subtree(operation, left_subtree_obj, right_subtree_obj, level):

    if level == 0 and left_subtree_obj is None and right_subtree_obj is None:
        return True # TODO: any validation steps for leaf nodes ie schema items inlcuding constants
    
    assert level > 0, f"{level=} and {left_subtree_obj=} {right_subtree_obj=}"

    get_op_type = operations_type_map[operation.strip().lower()]

    # TODO : validate the operation with the left and right subtree objects
    pass

def add_kth_level_subtrees(kth_level_nodes, level):
    """
    Add kth level nodes to kth_level_subtrees
    """
    # prepare kth level nodes from kth_level_nodes
    # input is text format node , example: [NODE_BEGIN]greater than or equal to[NB]st2[NB]st4[NODE_END]

    kth_level_subtrees.append([])

    for idx, node in enumerate(kth_level_nodes):

        lchild_count = node.count(LCHILD_NODE_TOKEN)
        rchild_count = node.count(RCHILD_NODE_TOKEN)

        child_token_count = lchild_count + rchild_count
        assert child_token_count <=2, f"Error in node {node}. More than 2 child tokens found"
        
        IS_NODE_BINARY = (child_token_count == 2)
        IS_NODE_UNARY = (child_token_count == 1)
        

        if IS_NODE_BINARY:
            assert lchild_count == 1 and rchild_count == 1, f"Error in node {node}. {lchild_count=} {rchild_count=}"
            operation, left_subtree, right_subtree = re.match(binary_node_regex, node).groups()
            assert operation is not None and operation in operations, f"Error in node {node} at {idx=} , {operation=}. Operation not found or None"
            assert left_subtree is not None, f"Error in node {node} at {idx=}. Left subtree not found"
            assert right_subtree is not None, f"Error in node {node} at {idx=}. Right subtree not found"
            assert level > 0, f"RAT operator at {level=} "

        elif IS_NODE_UNARY:
            assert lchild_count == 1, f"Error in node {node}. {lchild_count=} {rchild_count=}"
            operation, left_subtree = re.match(unary_node_regex, node).groups()
            assert operation is not None and operation in operations, f"Error in node {node} at {idx=}. Operation not found or None"
            assert left_subtree is not None, f"Error in node {node} at {idx=}. Left subtree not found"
            assert level > 0, f"RAT operator at {level=} "

        else:
            schema_item = re.match(leaf_node_regex, node).groups()
            assert schema_item is not None, f"Error in node {node}. Schema item not found"
            assert level == 0, f"{level=}"

        left_subtree_obj, right_subtree_obj = get_previous_level_node_objects(left_subtree, right_subtree, level)

        IS_VALID = validate_subtree(operation, left_subtree_obj, right_subtree_obj, level)

        if IS_VALID:
            if IS_NODE_BINARY:
                kth_level_subtrees[-1].append(Node(operation, children = [left_subtree_obj, right_subtree_obj], type=operations_type_map[operation.strip().lower()], level=level))
            elif IS_NODE_UNARY:
                kth_level_subtrees[-1].append(Node(operation, children=[left_subtree_obj], type=operations_type_map[operation.strip().lower()], level=level))
            else:
                kth_level_subtrees[-1].append(Node(schema_item), type="schema_constant", level=level)


def get_subtree_embeddings(traversed_tree):
    
    inputs = subtree_embed_tokenizer(traversed_tree, return_tensors="pt", padding=True).to(device)
    
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                 output_hidden_states=True)
    
    outputs = outputs.hidden_states[-1]

    level_node_embeddings[-1].append(outputs)

def create_kth_level_subtree_embeddings():

    # get current level nodes, if node present in kth_level_subtrees[-1] then it is a validated subtree node and can be passed for next level generation
    current_subtree_nodes = kth_level_subtrees[-1] 
    level_node_embeddings.append([]) # to store embeddings of nodes at this level

    for n in current_subtree_nodes:
        post_order = get_tree_traversal(n)[0] # to use for pre, in order also use get_tree_traversal(n, order=['post', 'pre', 'in'])

        # post order is a text format of the tree traversal, example: [NODE_BEGIN][NODE_BEGIN]age[NODE_END][NODE_BEGIN]60[NODE_END]greater than or equal to[NODE_END]
        get_subtree_embeddings(post_order)


def remove_keep(node):
    if node.name == "keep":
        node.children[0].parent = node.parent
        node.parent = None
        # node.parent.children.remove(node)

    else:
        for child in node.children:
            remove_keep(child)
    return node


def get_tree_traversal(root_node, order=['post', 'pre', 'in']):
    """
    root_node: node object for root
    order: list of traversals to be done in order
    """
    
    assert root_node is not None
    # remove keep (to remove duplicate node reps)
    root_node = remove_keep(root_node)

    result_str = ""
    # result_hashes = []

    def recurse_post(node):
        nonlocal result_str
        # nonlocal result_hashes
        result_str += NODE_BEGIN_TOKEN
        if not node.children:
            result_str += str(node.name)
        else:
            for child in node.children:
                recurse_post(child)
            result_str += str(node.name)
        result_str += NODE_END_TOKEN
        # result_hashes.append(node.hash)
    
    def recurse_pre(node):
        nonlocal result_str
        result_str += NODE_BEGIN_TOKEN
        if not node.children:
            result_str += str(node.name)
        else:
            result_str += str(node.name)
            for child in node.children:
                recurse_pre(child)
        result_str += NODE_END_TOKEN
    
    def recurse_in(node):
        nonlocal result_str
        result_str += NODE_BEGIN_TOKEN
        if not node.children:
            result_str += str(node.name)
        else:
            if len(node.children) == 1:
                recurse_in(node.children[0])
                result_str += str(node.name)
            else:
                for i, child in enumerate(node.children):
                    if i == 0:
                        recurse_in(child)
                    else:
                        result_str += str(node.name)
                        recurse_in(child)
        result_str += NODE_END_TOKEN

    traversal_order = {'pre': recurse_pre,
                       'in': recurse_in,
                       'post': recurse_post}
    
    traversals = []
    
    for o in order:
        result_str = ""
        traversal_order[o](root_node)
        traversals.append(result_str)

    #  TODO: explore pre and in order use

    return traversals

def generate_rat(input_with_schema):
    
    for k in range(0, T):

        kth_level_nodes = generate_kth_level_nodes(level=k, input_with_schema=input_with_schema) # kth level nodes in text format, schema is ranked schema (ie subset schema with highest cross encoder score)

        nodes = [(x + NODE_END_TOKEN) for x in kth_level_nodes.split(NODE_END_TOKEN) if x != ""] # kth level invidual nodes list with each node wrapped in node_begin and node_end token

        add_kth_level_subtrees(nodes, level=k)

        create_kth_level_subtree_embeddings()


    


def get_preds(input_path, schema_path, output_path, batch_size=1):

    assert batch_size == 1, f"Batch size should be equal to 1, but {batch_size=}"

    with open(input_path, 'r') as file:
        data = json.load(file)

    # with open(schema_path, 'r') as file:
    #     schemas = json.load(file) # TODO: update schema with complete schema (ie not null and foreign keys etc.)

    json_data = []

    for i in range(0, len(data), batch_size): 

        print(f"# {i}")

        batch = data[i:i + batch_size]

        ids, questions, sqls, db_ids = zip(*[(q["id"], q["model_input"], q["SQL"], q["db_id"]) for q in batch])

        # schemas = [schemas[d_id] for d_id in db_ids]

        # input_with_ranked_schemas = [cross_encoder(question, schema) for question, schema in zip(questions, schemas)] # TODO: implement cross encoder

        rat_pred = generate_rat(questions) # TODO: in progress
        # print("============new batch====================")

        json_data.extend([{"question_id": ids[i], "db_id": db_ids[i], "question": questions[i], "gold_SQL": sqls[i],
                           "pred_SQL": post_process_rat_to_sql(rat_pred[i])} for i in range(len(rat_pred))])  # TODO: implement post process RAT to SQL

        # save predicitons to file every DUMP_INTERVAL iterations
        if (i) % DUMP_INTERVAL == 0:
            with open(output_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
            # print(f"JSON dumped to file at iteration {i + 1}")

    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)
        

        


def main():

    # TODO: repeat for spider and bird dataset
    input_path = os.path.join('/root/CS726/data/spider/rat_input_data/dev_schema_subsetting_full_schema.json')
    output_path = os.path.join('/root/CS726/data/spider/processed/dev_preds.json')
    schema_path = os.path.join('/root/CS726/data/spider/processed/schemas.json')

    get_preds(input_path, schema_path, output_path)

if __name__ == "__main__":
    main()