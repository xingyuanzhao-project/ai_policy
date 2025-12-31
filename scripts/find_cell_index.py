import json

nb_path = r'c:\Users\meier\OneDrive\Documents\ai_policy\op6_def_plotting.ipynb'
target_string = "state_stats.columns"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if target_string in source:
            print(f"Found in cell index: {i}")
            print("Source snippet:")
            print(source)
            break
