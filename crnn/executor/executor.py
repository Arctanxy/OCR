import os
import sys
import torch
from tqdm import tqdm
from model.crnn import CRNN
from model.drrn import DRRN
from metric.acc import acc
from text_data.data import get_train_loader, prepare, to_lmdb, get_val_loader
from torch.utils.tensorboard import SummaryWriter

class Executor:
    def __init__(self, args):
        self.args = args
        prepare(args)
        # to_hdf5(args)
        if not os.path.exists(os.path.join(args.lmdb_path, "data.mdb")):
            to_lmdb(args)
        self.train_loader = get_train_loader(args)
        self.val_loader = get_val_loader(args)
        self.device = torch.device(args.device)

        self.model = CRNN(32, 1, 87, 512)
        self.model.to(self.device)
        self.sr = DRRN()
        self.sr.to(self.device)

        self.loss_func = torch.nn.CTCLoss(blank=86, reduction='mean',zero_infinity=True)
        self.acc_func = acc()
        self.optimizer = torch.optim.Adam([{"params":self.model.parameters()}, {"params":self.sr.parameters()}], lr = args.lr)

        self.logger = SummaryWriter(os.path.join(self.args.log_dir, self.args.task_name))

    def find_premodel(self):
        newest_time = 0
        newest_model = ""
        for f in os.listdir(self.args.model_dir):
            ctime = os.path.getctime(os.path.join(self.args.model_dir, f))
            if ctime > newest_time:
                newest_model = os.path.join(self.args.model_dir, f)
                newest_time = ctime
        print("found model: ", newest_model)
        return newest_model

    def save_model(self, name):
        torch.save({"crrn":self.model.state_dict(),"drrn":self.sr.state_dict()}, f"./ckpt/{name}.pth")

    def load_model(self, ckpt):
        params = torch.load(ckpt)
        self.model.load_state_dict(params["crrn"])
        self.sr.load_state_dict(params["drrn"])

    def train_one_epoch(self, epoch):
        epoch_loss_cls = []
        epoch_loss_sr = []
        epoch_accuracy = []
        for i, (img, blur_img, meta) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            # try:
            img = img.to(self.device)
            blur_img = blur_img.to(self.device)
            out_img = self.sr(blur_img)
            loss_sr = torch.nn.L1Loss()(out_img, img)
            # ===========

            label_id = meta["label_id"].to(self.device)
            length = meta["length"].to(self.device)
            out, feature = self.model(img) # ????????????
            out_blur, feature_blur = self.model(out_img)
            loss_feature = torch.nn.MSELoss()(feature, feature_blur)
            # ???????????????
            # if epoch > 1:
            #     out = self.model(out_img)
            out = torch.nn.functional.log_softmax(out, dim = -1)
            input_lengths = torch.full(size=(out.shape[1],), fill_value=out.shape[0], dtype=torch.long)
            self.optimizer.zero_grad()
            loss_cls = self.loss_func(out, label_id, input_lengths, length)
            loss = loss_sr  + loss_cls + loss_feature
            loss.backward()
            self.optimizer.step()
            epoch_loss_cls.append(loss_cls.item())
            epoch_loss_sr.append(loss_sr.item())
            accuracy = self.acc_func(out, label_id, length)
            epoch_accuracy.append(accuracy)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1,norm_type=2)
            if (i + 1) % 500 == 0:
                print("loss_cls {} loss_sr {} accuracy {}".format(sum(epoch_loss_cls) / len(epoch_loss_cls),
                                                                  sum(epoch_loss_sr) / len(epoch_loss_sr),
                                                                  sum(epoch_accuracy) / len(epoch_accuracy)))
                self.logger.add_scalar("epoch_loss_cls", sum(epoch_loss_cls) / len(epoch_loss_cls),
                                       epoch * len(self.train_loader) + i)
                self.logger.add_scalar("epoch_loss_sr", sum(epoch_loss_sr) / len(epoch_loss_sr),
                                       epoch * len(self.train_loader) + i)
                self.logger.add_scalar("epoch_accuracy", sum(epoch_accuracy) / len(epoch_accuracy),
                                       epoch * len(self.train_loader) + i)

    def train(self, num_epochs):
        premodel = self.find_premodel()
        if premodel:
            self.load_model(premodel)
        # self.validation(-1)
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)
            self.validation(epoch)
            self.save_model(epoch)

    def validation(self, epoch):
        self.model.eval()
        with torch.no_grad():
            epoch_accuracy = []
            for i, (img, blur_img, meta) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                # try:
                img = img.to(self.device)
                out_img = self.sr(img)
                label_id = meta["label_id"].to(self.device)
                # # ????????????????????????????????????target?????? ?????????
                if len(label_id) == 0:
                    continue
                length = meta["length"].to(self.device)
                _,_,h,w = img.shape
                if h >= w:
                    continue
                out = self.model(img)
                # out = self.model(out_img)
                out = torch.nn.functional.log_softmax(out, dim=-1)
                accuracy = self.acc_func(out, label_id, length)
                epoch_accuracy.append(accuracy)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                # except Exception as e:
                #     print("Error : ", e)
            print("validation accuracy {}".format(sum(epoch_accuracy) / len(epoch_accuracy)))
            self.logger.add_scalar("validation accuracy", sum(epoch_accuracy) / len(epoch_accuracy),
                                   epoch * len(self.train_loader))
        self.model.train()