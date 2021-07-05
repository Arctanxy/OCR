from glob import glob 
import pandas as pd 
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np 
from tqdm import tqdm


class TianChiData(Dataset):
    def __init__(self, subset="train"):
        self.img_list = sorted(glob("train/*jpg"),key = lambda x:int(os.path.basename(x).split(".")[0])
        ) + \
        sorted(glob("train1/*jpg"),key = lambda x:int(os.path.basename(x).split(".")[0])
        ) + \
        sorted(glob("train2/*jpg"),key = lambda x:int(os.path.basename(x).split(".")[0])
        )
            
        print(len(self.img_list))
        self.label_df = [
            pd.read_csv("img_list/Xeon1OCR_round1_train_20210524.csv"),
            pd.read_csv("img_list/Xeon1OCR_round1_train1_20210526.csv"),
            pd.read_csv("img_list/Xeon1OCR_round1_train2_20210526.csv")
        ]
        self.lengths = [self.label_df[0].shape[0], self.label_df[1].shape[0], self.label_df[2].shape[0]]
        self.labels = []
        set_id = 0
        self.label_list = []
        for i, img in enumerate(self.img_list):
            index = int(os.path.basename(img).split(".")[0])
            label = self.label_df[set_id][self.label_df[set_id]["数据ID"] == index]["融合答案"].values[0]
            label = eval(label)
            boxes = []
            texts = []
            for lab in label[0]:
                box = list(map(float, lab["coord"]))
                boxes.append(box)
                text = eval(lab["text"])["text"]
                texts.append(text)

            self.label_list.append({
                "boxes":boxes,
                "texts":texts,
                "orient":label[1]
            })
            if index == self.lengths[set_id] - 1:
                set_id += 1

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        # image = cv2.imread(self.img_list[index])
        label = self.label_list[index]
        return image, label

    def __len__(self):
        return len(self.label_list)

    
# 统计一下方向类别
def directions(data):
    from collections import defaultdict
    from tqdm import tqdm
    count = defaultdict(int)
    for i, (img, label) in tqdm(enumerate(data),total=len(data)):
        direction = label["orient"]["option"]
        count[direction] += 1
    print(count)
    # {'底部朝下': 11049, '底部朝左': 492, '底部朝右': 309, '底部朝上': 22}
    


if __name__ == "__main__":
    import sys
    from image import rotate_cut_img
    st = int(sys.argv[1])
    ed = int(sys.argv[2])  # max 11872
    if not os.path.exists("./rec_img"):
        os.makedirs("./rec_img")
    data = TianChiData()
    # directions(data)
    # exit(1)
    f = open("rec_label.txt", 'w', encoding="utf-8")
    idx = 0
    for i, (img, label) in tqdm(enumerate(data), total = len(data)):
        if i < st or i >= ed:
            continue
        width, height = img.size
        if label["orient"]["option"] == "底部朝左":
            img = img.transpose(Image.ROTATE_90)
        elif label["orient"]["option"] == "底部朝右":
            img = img.transpose(Image.ROTATE_270)
        elif label["orient"]["option"] == "底部朝上":
            img = img.transpose(Image.ROTATE_180)

        for j, box in enumerate(label["boxes"]):
            if label["orient"]["option"] == "底部朝左":
                # [x1, y1], [x2, y2], [x3, y3], [x4, y4]
                # [1, 2], [3, 4], [5, 6], [7, 8]
                # --> [y1, width - x], 
                box = [box[1], width - box[0], box[3], width - box[2], box[5], width - box[4], box[7], width - box[6]]
            elif label["orient"]["option"] == "底部朝右":
                box = [height - box[1], box[0], height - box[3], box[2], height - box[5], box[4], height - box[7], box[6]]
            elif label["orient"]["option"] == "底部朝上":
                box = [width - box[0], height - box[1], width - box[2], height - box[3], width - box[4], height - box[5], width - box[6], height - box[7]]
                print("index ", idx)
            try:
                partImg,box = rotate_cut_img(img,box)
                partImg.save(f"rec_img/{idx}.jpg")
                f.write(str(idx) + "\t" + label["texts"][j] + "\n")
            except:
                pass
            idx += 1
        
# 文本截取代码参考https://github.com/chineseocr/chineseocr/blob/app/apphelper/image.py
# todo：先旋转image和box，再截图
# 参考 https://blog.csdn.net/fanzonghao/article/details/86609090
# 详见utils.py