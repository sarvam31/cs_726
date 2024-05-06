# Format:
# Level 0: Q | RS -> [node_begin](<value>,<node_phi>,<node_phi>)[node_end] | ...
# Level 1: Q | RS | <node_embeddings> -> [node_begin](<op>,[node_1],[node_2])[node_end] | [node_begin](<op>,[node_3],[node_phi])[node_end] | ...
# Level 2: Q | RS | <node_embeddings> -> [node_begin](<op>,[node_1],[node_2])[node_end] | [node_begin](<op>,[node_3],[node_4])[node_end] | ...
# .
# .
# .
# Level 9: Q | RS | <node_embeddings> -> [node_begin](<op>,[node_1],[node_2])[node_end]
import sys
import copy
import traceback

# os.environ['HF_HOME'] = '/mnt/nas/sarvam/open_llm/huggingface_models/'
sys.path.insert(1, "/root/CS726/")

from anytree import Node, RenderTree, LevelOrderIter
import json
from mo_sql_parsing import parse
from constants import *

import pandas as pd
from random import sample
from utils import ra_postproc


from misc.ra_preproc import ast_to_ra


# with open('/root/repo/CS726/parsed/bird_dev_parsed.json', 'r') as f:
#     bird_dev = pd.DataFrame(json.load(f))


# sql = "SELECT T1.countryId ,  T1.CountryName FROM Countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country GROUP BY T1.countryId HAVING count(*)  >  3 UNION SELECT T1.countryId ,  T1.CountryName FROM Countries AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country JOIN MODEL_LIST AS T3 ON T2.Id  =  T3.Maker WHERE T3.Model  =  'fiat'"
# sql = "select count(*) from employees"


def balance_tree(node, ht):
    """
    Balances the tree by adjusting the height of the nodes.

    Args:
        node (Node): The root node of the tree.
        ht (int): The desired height of the tree.

    Returns:
        Node: The balanced tree with the specified height.
    """

    if node.height == ht:
        if node.children:
            for child in node.children:
                balance_tree(child, ht - 1)
        return node
    else:
        keep_node = Node("keep", parent=node.parent, children=[node])
        keep_node.n_type = node.n_type
        node.parent = keep_node
        balance_tree(node, ht - 1)


def fix_height(node, ht):
    """
    Fixes the height of the tree to the specified height.
    """
    while node.height != ht:
        node = Node("keep", children=[node], n_type=node.n_type)
    return node


def remove_literal(node):
    """
    Removes the literal nodes from the tree.
    """
    if node.name == "literal":
        node.children[0].parent = node.parent
        node.parent = None
        # node.parent.children.remove(node)

    else:
        for child in node.children:
            remove_literal(child)
    return node


# mapping from node name to human readable name
rename_nodes_map = {
    "Val_list": "constant union",
    "And": "and",
    "Or": "or",
    "gte": "greater than or equal to",
    "lte": "less than or equal to",
    "gt": "greater than",
    "lt": "less than",
    "like": "like",
    "not_like": "not like",
    "In": "in",
    "NotIn": "not in ",
    "Selection": "selection",
    "Project": "projection",
    "eq": "equal to",
    "neq": "not equal to",
    "Limit": "limit",
    "max": "max",
    "min": "min",
    "Orderby_desc": "order by descending",
    "Groupby": "group by",
    "avg": "average",
    "distinct": "distinct",
    "Orderby_asc": "order by ascending",
    "count": "count",
    "Product": "cartesian product",
    "in": "in",
    "sum": "sum",
    "intersect": "intersection",
    "except": "difference",
    "keep": "keep",
    "Distinct": "distinct",
}


# def extract_nodes_at_level(
#     nodes, level, nodes_prev_lvl_order=None, node_phi="[node_phi]", shuffle=False
# ):
#     nodes_repr = []
#     nodes_order = {}
#     if level == 0:
#         for i, node in enumerate(nodes[level]):
#             nodes_repr.append((node.val, node_phi, node_phi, node.n_type))
#             nodes_order[hash(node)] = i
#     else:
#         assert nodes_prev_lvl_order is not None
#         for i, node in enumerate(nodes[level]):
#             nodes_repr.append(
#                 (
#                     node.name,
#                     nodes_prev_lvl_order[hash(node.children[0])],
#                     (
#                         nodes_prev_lvl_order[hash(node.children[1])]
#                         if len(node.children) > 1
#                         else node_phi
#                     ),
#                     node.n_type
#                 )
#             )
#             nodes_order[hash(node)] = i
#     if shuffle:
#         new_order = sample(range(len(nodes_repr)), len(nodes_repr))
#         rev_hash = {v: k for k, v in nodes_order.items()}
#         nodes_repr_ = []
#         nodes_order_ = {}
#         for i_, i in enumerate(new_order):
#             nodes_repr_.append(nodes_repr[i])
#             nodes_order_[rev_hash[i]] = i_
#         return nodes_repr_, nodes_order_
#     return nodes_repr, nodes_order


def get_node_names(node):
    if node.children:
        return (
            node.name
            + " || "
            + " || ".join([get_node_names(child) for child in node.children])
        )
    else:
        return node.name


def update_node_names(node):
    if node.children:
        if node.name in ["Value", "Table"]:
            node.name = node.val
        else:
            node.name = rename_nodes_map.get(node.name, node.name)
            for child in node.children:
                update_node_names(child)
    else:
        if node.name in ["Value", "Table"]:
            node.name = node.val
        else:
            node.name = rename_nodes_map.get(node.name, node.name)
    return node


def add_level_wise_index(root):
    """
    Adds level wise index to the nodes of the tree.
    """
    for h in range(0, root.height + 1):
        idx = 0
        for k, v in enumerate(LevelOrderIter(root, filter_=lambda n: n.height == h)):

            v.level_idx = idx
            idx += 1

    return root


def print_tree(root):
    for pre, fill, node in RenderTree(root):
        print("%s%s%s" % (pre, node.name, node.level_idx))


# def get_node_repr(parsed):
#     root = ast_to_ra(parsed)
#     root = remove_literal(root)
#     root = balance_tree(root, root.height)
#     root = fix_height(root, 9)

#     foo = get_node_names(root)
#     # convert nested lists to a single list
#     print(set(foo.split(" || ")))


#     print_tree(root)


#     nodes = {}
#     for h in range(0, root.height + 1):
#         nodes[h] = {
#             v: k
#             for k, v in enumerate(
#                 LevelOrderIter(root, filter_=lambda n: n.height == h)
#             )
#         }
#     nodes_repr = {}
#     flag_shuffle = True
#     for i in range(root.height + 1):
#         if i == 0:
#             nodes_repr[0] = extract_nodes_at_level(nodes, 0, shuffle=flag_shuffle)
#         else:
#             nodes_repr[i] = extract_nodes_at_level(
#                 nodes, i, nodes_repr[i - 1][1], shuffle=flag_shuffle
#             )
#     return nodes_repr


# def get_node_repr_training(node_tuple):
#     node_repr = '[NODE_BEGIN]'
#     node_repr += str(node_tuple[0])
#     if node_tuple[1] != '[node_phi]':
#         node_repr += '[LEFT_CHILD]st' + str(node_tuple[1])
#     if node_tuple[2] != '[node_phi]':
#         node_repr += '[RIGHT_CHILD]st' + str(node_tuple[2])
#     node_repr += '[NODE_END]'
#     return node_repr


# parsed_sql = parse(sql)


# nodes_repr = get_node_repr(parsed_sql)


# def gen_training_repr(nodes_repr):
#     training_repr = []
#     for k, v in nodes_repr.items():
#         training_repr.append(
#             "".join([get_node_repr_training(node_tuple) for node_tuple in v[0]])
#         )
#     return training_repr


# training_repr = gen_training_repr(nodes_repr)


# for i, level in enumerate(training_repr):
#     print(f"Level {i}:")
#     print(level)


# with open("/root/CS726/data/spider/processed/train_gold.json", "r") as f:
#     data = json.load(f)

# bar = 0
# bar2 = 0
# total = 0
# aa = []


def remove_keep(node):
    """
    Removes the keep nodes from the tree.
    """
    if node.name == "keep":
        node.children[0].parent = node.parent
        node.parent = None
        # node.parent.children.remove(node)
    else:
        for child in node.children:
            remove_keep(child)
    return node


def get_tree_traversal(root_node, order=["post", "pre", "in"]):
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

    traversal_order = {"pre": recurse_pre, "in": recurse_in, "post": recurse_post}

    traversals = []

    for o in order:
        result_str = ""
        traversal_order[o](root_node)
        traversals.append(result_str)

    #  TODO: explore pre and in order use

    return traversals


# def get_subtree_embeddings(traversed_tree):

#     inputs = subtree_embed_tokenizer(traversed_tree, return_tensors="pt", padding=True).to(device)

#     outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
#                                  output_hidden_states=True)

#     return outputs.hidden_states[0]

#     # level_node_embeddings[-1].append(outputs)


def generate_input_output(root, j_obj):
    """
    Generates the input and output sequences for the model.
    """
    input_seq = ""
    output_seq = ""
    # traversals = ""
    st_list = []
    previous_level_subtrees = []
    # previous_level_embeddings = []

    for h in range(0, root.height + 1):

        input_subtree_traversal_list = []
        output_seq = ""
        input_seq = ""
        st_list = []

        for i in previous_level_subtrees:
            t = copy.deepcopy(i)
            tree_trav = get_tree_traversal(t)

            input_subtree_traversal_list.append(
                {
                    f"st{i.level_idx}": {
                        "post": tree_trav[0],
                        "pre": tree_trav[1],
                        "in": tree_trav[2],
                    }
                }
            )
            # post_order = "postorder traversal: " + tree_trav[0]
            # pre_order = "preorder traversal: " + tree_trav[1]
            # in_order = "inorder traversal: " + tree_trav[2]

            # traversals = post_order + "||" + pre_order + "||" + in_order

            # pot_embed = get_subtree_embeddings(post_order)
            # previous_level_embeddings.append(post_embed)
            st_list.append(f"st{i.level_idx}")

        if len(st_list) > 0:
            input_seq = "|".join(st_list)

        previous_level_subtrees = []

        for k, v in enumerate(LevelOrderIter(root, filter_=lambda n: n.height == h)):

            num_children = len(v.children)

            assert num_children <= 2, f"Expected <=2 children, {num_children=}"

            if num_children == 2:
                output_seq += (
                    NODE_BEGIN_TOKEN
                    + str(v.name)
                    + LCHILD_NODE_TOKEN
                    + f"st{v.children[0].level_idx}"
                    + RCHILD_NODE_TOKEN
                    + f"st{v.children[1].level_idx}"
                    + NODE_END_TOKEN
                )

            elif num_children == 1:
                output_seq += (
                    NODE_BEGIN_TOKEN
                    + str(v.name)
                    + LCHILD_NODE_TOKEN
                    + f"st{v.children[0].level_idx}"
                    + NODE_END_TOKEN
                )

            else:
                output_seq += NODE_BEGIN_TOKEN + str(v.name) + NODE_END_TOKEN

            previous_level_subtrees.append(v)

        previous_level_subtrees.sort(key=lambda x: x.level_idx)

        if input_seq != "":
            model_input = j_obj["input_sequence"] + " | " + input_seq
        else:
            model_input = j_obj["input_sequence"]

        training_data.append(
            {
                "id": j_obj["id"],
                "db_id": j_obj["db_id"],
                "input_sequence": j_obj["input_sequence"],
                "SQL": j_obj["SQL"],
                "model_input": model_input,
                "output": output_seq,
                "input_subtree_traversal_list": input_subtree_traversal_list,
            }
        )


training_data = []
total = 0
success = 0
fail = 0

with open(
    "/root/CS726/data/spider/resdsql_subsetting_schema/train_schema_subsetting_full_schema.json",
    "r",
) as f:
    data = json.load(f)

for i in data:
    total += 1
    try:
        if "+ " not in i["SQL"] and "- " not in i["SQL"]:
            root = ast_to_ra(parse(i["SQL"]))
            root = remove_literal(root)
            root = balance_tree(root, root.height)
            root = fix_height(root, 9)
            root = update_node_names(root)
            root = add_level_wise_index(root)

            generate_input_output(root, i)
            success += 1

            # print_tree(root)
            # foo = get_node_names(root)
            # for x in foo.split(" || "):
            #     aa.append(x)
            #     if x in ['add', 'sub']:
            #         print(i['SQL'])

    except Exception as e:
        # print(e)
        # print(traceback.format_exc())
        # print(i['SQL'])
        fail += 1

print(f"Total: {total}")
print(f"Passed: {success}")
print(f"Failed: {fail}")

with open(
    "/root/CS726/data/spider/rat_input_data/train_schema_subsetting_full_schema.json",
    "w",
) as f:
    json.dump(training_data, f, indent=4)


# dev k1 4 k2 4
# Total: 1034
# Passed: 877
# Failed: 157

# train k1 4 k2 4
# Total: 7000
# Passed: 5929
# Failed: 1045
