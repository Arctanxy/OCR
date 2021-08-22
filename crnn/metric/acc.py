import torch

class acc:
    def __call__(self, out, target, length):
        preds = torch.argmax(out, dim = -1)
        _, N = preds.shape
        st = 0
        single_word_accuracy = []
        for i in range(N):
            pred = self.decode(preds[:,i])
            target_item = target[st:st + length[i]]
            st += length[i]
            # # 删除一部分符号之后，可能target为空 的情况
            # if len(target_item) == 0:
            #     single_word_accuracy.append(0)
            jaccard = self.single_word_acc(pred, target_item)
            single_word_accuracy.append(jaccard)
        # print(preds[:,i], target_item)
        return sum(single_word_accuracy) / len(single_word_accuracy)

    def decode(self, pred):
        rlt = []
        rlt.append(pred[0])
        for i in range(1, len(pred)):
            if pred[i] != 86 and pred[i] != pred[i - 1]:
                rlt.append(pred[i])
        return torch.tensor(rlt)

    def single_word_acc(self, pred, target_item):
        intersection = 0
        target_set = target_item.cpu().data.numpy().tolist()
        for item in target_set:
            if (pred == item).sum() > 0:
                intersection += 1
        return intersection / len(target_set)



