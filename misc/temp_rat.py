from moz_sql_parser import parse
import json


sql = "SELECT name, age FROM Employees WHERE department = 'HR' AND title='Manager' AND age>40;"
parsed = parse(sql)
# print(parsed)
# print()
# print()

with open("/raid/infolab/smitj/workspace/tmp/CS726/data/spider/processed/dev_gold.json", "r") as f:
    data = json.load(f)

for i, item in enumerate(data):
    if i in [499, 796]:
        parsed = parse(item["SQL"])
        print(json.dumps(parsed, indent=4))
        print()
        print()

def calculate_height(tree):
    if not isinstance(tree, dict):
        return 0  # Non-dictionary items contribute no height
    
    max_depth = 0  # Start with a base depth of 0 for each node
    for key, value in tree.items():
        if isinstance(value, dict):
            # If the value is a dictionary, calculate its height recursively
            height = calculate_height(value)
        elif isinstance(value, list):
            # Calculate the height of each dictionary item in the list, return the max
            # Ensure there is at least one dictionary to evaluate, otherwise use 0 as default
            height = max((calculate_height(item) for item in value if isinstance(item, dict)), default=0)
        else:
            continue  # If it's neither dict nor list of dicts, continue without adjusting height
        
        max_depth = max(max_depth, 1 + height)  # Compare current max depth with new depth

    return max_depth + 1  # Include this node's own level in the count

# Given JSON tree
json_tree = {
    'select': [{'value': 'name'}, {'value': 'age'}],
    'from': 'Employees',
    'where': {
        'and': [
            {'eq': ['department', {'literal': 'HR'}]},
            {'eq': ['title', {'literal': 'Manager'}]},
            {'gt': ['age', 40]}
        ]
    }
}

# Calculate the height of the tree
height = calculate_height(json_tree)
print("The height of the JSON tree is:", height)

height = calculate_height(parsed)
print("The height of the JSON tree is:", height)

print(json.dumps(parse("SELECT schools.zip FROM schools JOIN frpm ON schools.cdscode = frpm.cdscode WHERE frpm.\"charter school (y/n)\" = 1 AND schools.county = 'Fresno' AND schools.district = 'Fresno county office of Education'"), indent=4))
height = calculate_height(parse("SELECT schools.zip FROM schools JOIN frpm ON schools.cdscode = frpm.cdscode WHERE frpm.\"charter school (y/n)\" = 1 AND schools.county = 'Fresno' AND schools.district = 'Fresno county office of Education'"))
print("The height of the JSON tree is:", height)