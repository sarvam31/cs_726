from pathlib import Path
import torch

device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

T = 9
DUMP_INTERVAL = 10

NODE_BEGIN_TOKEN = "[NODE_BEGIN]"
NODE_END_TOKEN = "[NODE_END]"
LCHILD_NODE_TOKEN = "[LEFT_CHILD]"
RCHILD_NODE_TOKEN = "[RIGHT_CHILD]"

NUM_ST_NODES = 50
unary_node_regex = r"^\[NODE_BEGIN\]([a-z ]+)\[LEFT_CHILD\](st\d+)\[NODE_END\]$"
binary_node_regex = r"^\[NODE_BEGIN\]([a-z ]+)\[LEFT_CHILD\](st\d+)\[RIGHT_CHILD\](st\d+)\[NODE_END\]$"
leaf_node_regex = r"^\[NODE_BEGIN\]([a-z ]+)\[NODE_END\]$"

operations = ["union", "intersection", "difference", "selection", "cartesian product", "projection", "and", "or", "greater than", "greater than or equal to", "less than", "less than or equal to", "equal to", "not equal to", "constant union", "order by ascending", "order by descending", "group by", "limit", "in", "not in ", "like", "not like", "sum", "max", "min", "count", "average", "distinct", "keep"]

operation_types = ["predicate", "relation", "schema_constant", "any"]

operations_type_map = { 
                        "union": "relation",
                        "intersection": "relation",
                        "difference": "relation",
                        "selection": "relation",
                        "cartesian product": "relation",
                        "projection": "relation",
                        "and": "predicate",
                        "or": "predicate",
                        "greater than": "predicate",
                        "greater than or equal to": "predicate",
                        "less than": "predicate",
                        "less than or equal to": "predicate",
                        "equal to": "predicate",
                        "not equal to": "predicate",
                        "constant union": "schema_constant",
                        "order by ascending": "relation",
                        "order by descending": "relation",
                        "group by": "relation",
                        "limit": "relation",
                        "in": "predicate",
                        "not in ": "predicate",
                        "like": "predicate",
                        "not like": "predicate",
                        "sum": "schema_constant",
                        "max": "schema_constant",
                        "min": "schema_constant",
                        "count": "schema_constant",
                        "average": "schema_constant",
                        "distinct": "schema_constant",
                        "keep" : "any"
}

op_input_types = {
                    "union": ["relation", "relation"],
                    "intersection": ["relation", "relation"],
                    "difference": ["relation", "relation"],
                    "selection": ["predicate", "relation"],
                    "cartesian product": ["relation", "relation"],
                    "projection": ["schema_constant", "relation"],
                    "and": ["predicate", "predicate"],
                    "or": ["predicate", "predicate"],
                    "greater than": ["schema_constant", "schema_constant"],
                    "greater than or equal to": ["schema_constant", "schema_constant"],
                    "less than": ["schema_constant", "schema_constant"],
                    "less than or equal to": ["schema_constant", "schema_constant"],
                    "equal to": ["schema_constant", "schema_constant"],
                    "not equal to": ["schema_constant", "schema_constant"],
                    "constant union": ["schema_constant", "schema_constant"],
                    "order by ascending": ["schema_constant", "relation"],
                    "order by descending": ["schema_constant", "relation"],
                    "group by": ["schema_constant", "relation"],
                    "limit": ["schema_constant", "relation"],
                    "in": ["schema_constant", "relation"],
                    "not in ": ["schema_constant", "relation"],
                    "like": ["schema_constant", "schema_constant"],
                    "not like": ["schema_constant", "schema_constant"],
                    "sum": ["schema_constant"],
                    "max": ["schema_constant"],
                    "min": ["schema_constant"],
                    "count": ["schema_constant"],
                    "average": ["schema_constant"],
                    "distinct": ["schema_constant"],
                    "keep" : ["any"] 

}