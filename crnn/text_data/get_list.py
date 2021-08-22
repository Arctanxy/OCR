import re
import os
from glob import glob

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
        if os.path.isdir(f):
            find_images(f, img_list)
        elif os.path.isfile(f) and ("jpg" in f or "png" in f):
            img_list.append(f)

def load_syth90k(folder):
    img_list = []
    img_list = find_images(folder, img_list)
    labels = []
    for img in img_list:
        basename = os.path.basename(img).split(".")[0]
        label = basename.split("_")[1]
        labels.append(label)
    return img_list, labels