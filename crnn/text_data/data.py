import os
import re
import sys
import h5py
import lmdb
import torch
import numpy as np
from torchvision.transforms import ToTensor
from glob import glob
from tqdm import tqdm
from PIL import Image
from utils.tools import load_icdar15, load_syth90k
from torch.utils.data import DataLoader, Dataset
import pickle

def save_obj(obj, file_name):
    with open(f"./res/{file_name}.pkl", 'wb') as f:
        pickle.dump(obj, f)
    return None

def prepare(args):
    if os.path.exists("./res/train_imgs.pkl"):
        return 1
    print("preparing ... ")
    train_imgs, train_labels = load_syth90k(args.train_folder)
    val_imgs, val_labels = load_icdar15(args.val_folder)

    save_obj(train_imgs, "train_imgs")
    save_obj(train_labels, "train_labels")
    save_obj(val_imgs, "val_imgs")
    save_obj(val_labels, "val_labels")

    labels = train_labels   # ignore labels in validation dataset
    alphabet = []
    for item in labels:
        for char in item:
            if char not in alphabet:
                alphabet.append(char)
    with open("./res/alpha.txt", 'w') as f:
        for alpha in alphabet:
            f.write(alpha + "\n")
    return 0

def to_lmdb(args):
    if os.path.exists("./res/train_imgs.pkl"):
        with open("./res/train_imgs.pkl", 'rb') as f:
            train_img_list = pickle.load(f)
        with open("./res/train_labels.pkl", 'rb') as f:
            train_labels = pickle.load(f)
        with open("./res/val_imgs.pkl", 'rb') as f:
            val_img_list = pickle.load(f)
        with open("./res/val_labels.pkl", 'rb') as f:
            val_labels = pickle.load(f)
        with open("./res/alpha.txt", 'r') as f:
            alphabet = f.read().split("\n")[:-1]

    # 创建数据库文件
    env = lmdb.open(args.lmdb_path, max_dbs=5, map_size=int(1e10))
    # 创建对应的数据库
    train_data = env.open_db("train_data".encode())
    train_label = env.open_db("train_label".encode())
    train_length = env.open_db("train_length".encode())
    val_data = env.open_db("val_data".encode())
    val_label = env.open_db("val_label".encode())
    # 把图像数据写入到LMDB中
    num_samples = 100000
    with env.begin(write=True) as txn:
        for idx, path in tqdm(enumerate(train_img_list[:num_samples]),total=num_samples):
            # image
            try:
                img = Image.open(train_img_list[idx]).convert("L")
            except Exception as e:
                print("Error : ", e)
                continue
            w, h = img.size
            ratio = h / args.fix_h
            w_ = int(w / ratio)
            img = img.resize((w_, args.fix_h))
            img_array = np.array(img).astype(np.int16)
            img_array = img_array[np.newaxis, :, :]
            txn.put(str(idx).encode(), img_array, db=train_data)
            txn.put(str(idx).encode(), np.array(w_).astype(np.int16), db=train_length)
            label = train_labels[idx]
            label_id = []
            for char in label:
                if char in alphabet:
                    label_id.append(alphabet.index(char))
                else:
                    continue
            txn.put(str(idx).encode(), np.array(label_id).astype(np.int16), db=train_label)

        for idx, path in enumerate(val_img_list):
            # image
            try:
                img = Image.open(val_img_list[idx]).convert("L")
            except Exception as e:
                print("Error : ", e)
                continue
            w, h = img.size
            ratio = h / args.fix_h
            w_ = int(w / ratio)
            img = img.resize((w_, args.fix_h))
            img_array = np.array(img).astype(np.int16)
            img_array = img_array[np.newaxis, :, :]
            txn.put(str(idx).encode(), img_array, db=val_data)

        for idx, path in enumerate(val_labels):
            label_id = []
            for char in label:
                if char in alphabet:
                    label_id.append(alphabet.index(char))
                else:
                    continue
            txn.put(str(idx).encode(), np.array(label_id).astype(np.int16), db=val_label)

class Data(Dataset):
    def __init__(self, args, data_type = "train"):
        self.args = args
        self.fix_h = args.fix_h
        self.data_type = data_type
        if self.data_type == "train":
            with open("./res/train_imgs.pkl", 'rb') as f:
                self.img_list = pickle.load(f)
            with open("./res/train_labels.pkl", 'rb') as f:
                self.labels = pickle.load(f)
        elif self.data_type == "val":
            with open("./res/val_imgs.pkl", 'rb') as f:
                self.img_list = pickle.load(f)
            with open("./res/val_labels.pkl", 'rb') as f:
                self.labels = pickle.load(f)
        else:
            raise Exception("Data Type not supported. ")
        if os.path.exists("./res/alpha.txt"):
            with open("./res/alpha.txt", 'r') as f:
                self.alphabet = f.read().split("\n")[:-1]
        else:
            raise Exception("./res/alpha.txt not exist. ")

        if os.path.exists(os.path.join(args.lmdb_path, "data.mdb")):
            self.lmdb = True
        #     # env = lmdb.open(args.lmdb_path, max_dbs=4, map_size=int(1e12), readonly=True)
        #     self.env = lmdb.open("./res/lmdb", max_dbs=4, map_size=int(1e12), readonly=True)
        #     self.train_data = self.env.open_db("train_data".encode())
        #     self.train_label = self.env.open_db("train_label".encode())
        #     # self.train_length = self.env.open_db("train_length".encode()) # 暂时不用
        #     self.val_data = self.env.open_db("val_data".encode())
        #     self.val_label = self.env.open_db("val_label".encode())
        #     self.txn = self.env.begin()
        #     self._length = self.txn.stat(db=self.train_data)["entries"] if self.data_type == "train" else \
        #         self.txn.stat(db=self.val_data)[
        #             "entries"]
            # self.train_data = None
            # self.train_label = None
            # self.train_length = None # 暂时不用
            # self.val_data = None
            # self.val_label = None
            # self.txn = None
            # self._length = None

    def open_lmdb(self):
        env = lmdb.open("./res/lmdb", max_dbs=4, map_size=int(1e12), readonly=True)
        self.train_data = env.open_db("train_data".encode())
        self.train_label = env.open_db("train_label".encode())
        # self.train_length = env.open_db("train_length".encode()) # 暂时不用
        self.val_data = env.open_db("val_data".encode())
        self.val_label = env.open_db("val_label".encode())
        self.txn = env.begin()
        self._length = self.txn.stat(db=self.train_data)["entries"] if self.data_type == "train" else \
            self.txn.stat(db=self.val_data)[
                "entries"]


    def __getitem__(self, index):
        # if self.hf is None:
        if self.lmdb is False:
            img = Image.open(self.img_list[index]).convert("L")
            w, h = img.size
            ratio = h / self.fix_h
            w_ = int(w / ratio)
            img = img.resize((w_, self.fix_h))
            label = self.labels[index]
            label_id = self.to_id(label)
            meta = {"label":label, "label_id":label_id}
            img, meta = self.trans(img,meta)
        else:
            if not hasattr(self, "train_data"):
                self.open_lmdb()
            idx = str(index).encode()
            if self.data_type == "train":
                image = self.txn.get(idx, db=self.train_data)
                label = self.txn.get(idx, db=self.train_label)
            else:
                image = self.txn.get(idx, db = self.val_data)
                label = self.txn.get(idx, db = self.val_label)

            img = np.frombuffer(image, np.int16).reshape(1, self.fix_h, -1)
            img = torch.from_numpy(img) / 256.0
            label = np.frombuffer(label, np.int16)
            meta = {"label_id":label}
        return img, meta

    def to_id(self, label):
        indices = []
        for char in label:
            if char in self.alphabet:
                idx = self.alphabet.index(char)
                indices.append(idx)
            else:
                continue
        return indices

    def trans(self, img, meta):
        img = ToTensor()(img)
        return img, meta

    def __len__(self):
        # return self._length
        if self.data_type == "train":
            return 80000
        else:
            return len(self.img_list)

def collate_fn(batch):
    imgs = [item[0] for item in batch]
    metas = [item[1] for item in batch]
    max_length = 0
    fix_c, fix_h = imgs[0].shape[:2]
    for img in imgs:
        c, h, w = img.shape
        if w > max_length:
            max_length = w
    label_ids = [item["label_id"] for item in metas]
    out_mat = torch.zeros((len(imgs), fix_c, fix_h, max_length))
    out_mask = torch.zeros((len(imgs), fix_c, fix_h, max_length))
    out_label = []
    out_length = []
    for i,img in enumerate(imgs):
        w = img.shape[-1]
        out_mat[i,:,:,:w] = img
        out_mask[i,:,:,:w] = 1
        out_label.extend(label_ids[i])
        out_length.append(len(label_ids[i]))

    meta = {"mask": out_mask, "label_id": torch.tensor(out_label),"length":torch.tensor(out_length)}
    return out_mat, meta

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.lmdb = True
    env = lmdb.open("./res/lmdb", max_dbs=4, map_size=int(1e12), readonly=True)
    dataset.train_data = env.open_db("train_data".encode())
    dataset.train_label = env.open_db("train_label".encode())
    # self.train_length = env.open_db("train_length".encode()) # 暂时不用
    dataset.val_data = env.open_db("val_data".encode())
    dataset.val_label = env.open_db("val_label".encode())
    dataset.txn = env.begin()
    dataset._length = dataset.txn.stat(db=dataset.train_data)["entries"] if dataset.data_type == "train" else \
    dataset.txn.stat(db=dataset.val_data)[
        "entries"]

def get_train_loader(args):
    d = Data(args)
    dl = DataLoader(d, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True,
                    num_workers=4) #, worker_init_fn=worker_init_fn, multiprocessing_context="spawn")
    return dl

def get_val_loader(args):
    d = Data(args, data_type="val")
    dl = DataLoader(d, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=4)
    return dl

