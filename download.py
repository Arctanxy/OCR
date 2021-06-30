import os
import re
import json
from glob import glob 
import pandas as pd 


src_dir = "./img_list"
train_list = glob(f"{src_dir}/*train*csv")
test_list  = glob(f"{src_dir}/*test*csv")
print(train_list, test_list)

def gen_tasks(task_name):
    if not os.path.exists(task_name):
        os.makedirs(task_name)
    with open(f"{task_name}/run.sh", 'w') as f:
        f.write(f"nohup python download_sub.py {src_dir}/{task_name}.list {task_name} & ")


for lst in (train_list + test_list):
    if "train" in lst:
        df = pd.read_csv(lst)
        urls = [json.loads(item)["tfspath"] for item in df["原始数据"].values]
        indices = df["数据ID"].values
        task_name = re.findall(r"_(train\d*)_", lst)[0]
    else:
        df = pd.read_csv(lst)
        urls = [json.loads(item)["tfspath"] for item in df["原始数据"].values]
        indices = df["数据ID"].values
        task_name = re.findall(r"_(test\d+)_", lst)[0]

    with open(f"{src_dir}/{task_name}.list", 'w') as f:
        for i in range(len(indices)):
            f.write(f"{indices[i]}\t{urls[i]}\n")

    gen_tasks(task_name)


#! todo: finish download code 