import re
import numpy as np
import os
from glob import glob
from PIL import Image

def load_icdar15(folder):
    img_list = sorted(glob(os.path.join(folder, "*png")) + \
                      glob(os.path.join(folder, "*jpg")),
                      key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    with open(os.path.join(folder, "gt.txt"), encoding="gbk") as f:
        labels = f.read().split("\n")[:-1]
    labels = [re.findall(r'\"(.*?)\"', item.split('png,')[1])[0] for item in labels]
    return img_list, labels

def find_images(folder, img_list):
    for f in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, f)):
            find_images(os.path.join(folder, f), img_list)
        elif os.path.isfile(os.path.join(folder, f)) and ("jpg" in f or "png" in f):
            img_list.append(os.path.join(folder, f))
    print("found {} images".format(len(img_list)))

def load_syth90k(folder):
    train_list = os.path.join(folder, "annotation_train.txt")
    # val_list = os.path.join(folder, "annotation_val.txt")
    # test_list = os.path.join(folder, "annotation_test.txt")
    abs_path = []
    with open(train_list, 'r') as f:
        for line in f:
            img_path = line.strip().split()[0]
            path = os.path.join(folder, img_path)
            abs_path.append(path)
            if len(abs_path) % 1000000 == 0:
                print("loaded {} files".format(len(abs_path)))
    labels = [item.split("_")[1] for item in abs_path]
    return abs_path, labels
