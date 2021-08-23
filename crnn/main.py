import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type = str, default="cuda:0")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--val_folder", type=str, default=r"F:\scripts\blurOCR\icdar15_incident\ch4_training_word_images_gt")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--train_folder", type=str, default=r"C:\Users\Administrator\Downloads\mjsynth.tar\mjsynth\mnt\ramdisk\max\90kDICT32px")
parser.add_argument("--fix_h", type=int, default=32)
# parser.add_argument("--hdf5_path", type=str, default="./res/data.hdf5")
parser.add_argument("--lmdb_path", type=str, default="./res/lmdb")
parser.add_argument("--model_dir", type=str, default="./ckpt")
args = parser.parse_args()


if __name__ == "__main__":
    from executor.executor import Executor
    print(args)
    exec = Executor(args)
    exec.train(1000)

    # loss_func = torch.nn.CTCLoss(blank=86, reduction='mean',zero_infinity=False)
    # network = CRNN(32, 3, 87, 512).cuda()
    # d = data.get_train_loader(r"F:\scripts\blurOCR\icdar15_incident\ch4_training_word_images_gt")
    # for i, (img, meta) in enumerate(d):
    #     print(img.shape,meta.keys())
    #     out = torch.nn.functional.log_softmax(network(img.cuda()), dim = -1)
    #     print(out.shape)
    #     label_id = meta["label_id"]
    #     length = meta["length"]
    #     input_lengths = torch.full(size=(out.shape[1],), fill_value=out.shape[0], dtype=torch.long)
    #     loss = loss_func(out, label_id, input_lengths, length)
    #     print(loss.item())
    #     break
