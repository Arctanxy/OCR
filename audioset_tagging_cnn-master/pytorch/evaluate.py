from copy import deepcopy
import torch
import numpy as np 
from sklearn import metrics

from pytorch_utils import forward


class Evaluator(object):
    def __init__(self, model, num_classes = 100):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        self.num_classes = num_classes
        self.sum = np.array([0 for i in range(num_classes)])
        self.recall = np.array([0 for i in range(num_classes)])

    def evaluate(self, data_loader):
        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        target=target.squeeze()
        preds = np.argmax(clipwise_output, axis = -1)
        target = np.argmax(target, axis= -1)
        rlt = self.cal(target, preds)
        return rlt
        
    def cal(self, target, preds):
        self.sum = np.array([(target == i).sum() for i in range(self.num_classes)])
        self.recall = np.array([(preds[target == i] == target[target == i]).sum() for i in range(self.num_classes)])
        mask = self.sum != 0
        valid_recalls = self.recall[mask] / self.sum[mask]
        average_recall = np.mean(valid_recalls)
        statistics = {"average_recall":average_recall}
        return statistics
        # raise NotImplementedError




    def evaluate_bak(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        target=target.squeeze()
        average_precision = metrics.average_precision_score(target, clipwise_output, average=None)

        # auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        import pdb;pdb.set_trace()
        auc = 0
        statistics = {'average_precision': average_precision, 'auc': auc}

        return statistics