import os
import sys
from urllib.request import urlretrieve

file = sys.argv[1]
task_name = sys.argv[2]

folder = os.path.dirname(file)
info = {}
with open(file, 'r') as f:
    lines = f.read().split("\n")
    info = {}
    for line in lines:
        if "\t" not in line:
            continue
        k, v = line.split("\t")
        info[k] = v

for k,v in info.items():
    urlretrieve(v, f"{task_name}/{k}.jpg")
    print(f"saving to {task_name}/{k}.jpg")