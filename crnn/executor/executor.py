import os
import sys
import torch
from tqdm import tqdm
from model.crnn import CRNN
from metric.acc import acc
from text_data.data import get_train_loader, prepare, to_lmdb, get_val_loader

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
        self.loss_func = torch.nn.CTCLoss(blank=86, reduction='mean',zero_infinity=True)
        self.acc_func = acc()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr)

    def train_one_epoch(self, epoch):
        epoch_loss = []
        epoch_accuracy = []
        for i, (img, meta) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            # continue
            try:
                img = img.to(self.device)
                label_id = meta["label_id"].to(self.device)
                length = meta["length"].to(self.device)
                out = self.model(img.cuda())
                out = torch.nn.functional.log_softmax(out, dim = -1)
                input_lengths = torch.full(size=(out.shape[1],), fill_value=out.shape[0], dtype=torch.long)
                self.optimizer.zero_grad()
                loss = self.loss_func(out, label_id, input_lengths, length)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item())
                accuracy = self.acc_func(out, label_id, length)
                epoch_accuracy.append(accuracy)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1,norm_type=2)
                if (i + 1) % 1000 == 0:
                    print("loss {} accuracy {}".format(sum(epoch_loss) / len(epoch_loss), sum(epoch_accuracy) / len(epoch_accuracy)))
            except Exception as e:
                print("Error : ", e)

    def train(self, num_epochs):
        self.validation(-1)
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)
            self.validation(epoch)

    def validation(self, epoch):
        self.model.eval()
        with torch.no_grad():
            epoch_accuracy = []
            for i, (img, meta) in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                try:
                    img = img.to(self.device)
                    label_id = meta["label_id"].to(self.device)
                    length = meta["length"].to(self.device)
                    out = self.model(img.cuda())
                    out = torch.nn.functional.log_softmax(out, dim=-1)
                    accuracy = self.acc_func(out, label_id, length)
                    epoch_accuracy.append(accuracy)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                except Exception as e:
                    print("Error : ", e)
            print("validation accuracy {}".format(sum(epoch_accuracy) / len(epoch_accuracy)))
            torch.save(self.model.state_dict(), "./ckpt/{}.pth".format(epoch))
        self.model.train()