from glob import glob 
import pandas as pd 
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os

class OCRData(Dataset):
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
            # print(f"{index} \n {self.label_df[set_id]['数据ID']}")
            # print(i, index, set_id, self.lengths, len(self.img_list), sum(self.lengths))
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
                "texts":texts
            })
            if index == self.lengths[set_id] - 1:
                set_id += 1

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        label = self.label_list[index]
        return image, label

    def __len__(self):
        return len(self.label_list)


if __name__ == "__main__":
    from PIL import ImageDraw
    data = OCRData()
    for i, (img, label) in enumerate(data):
        draw = ImageDraw.Draw(img)
        for j, box in enumerate(label["boxes"]):
            draw.polygon(box, outline="red")            
            print(label["texts"][j])
            if j > 4:
                break
        img.show()
        break

        

