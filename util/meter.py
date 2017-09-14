import numpy as np

# TODO: double check the evaluation code, e.g., ignored class 0
class SegmentationMeter(object):
    def __init__(self, n_class, ignore_index=-100):
        self.n_class = n_class
        self.hist = np.zeros((n_class, n_class))
        self.ignore_index = ignore_index

    def update_confmat(self, label_true, label_pred):
        label_true = label_true.flatten()
        label_pred = label_pred.flatten()
        mask = (label_true >= 0) & (label_true < self.n_class) & (label_true != self.ignore_index)
        self.hist += np.bincount(
            self.n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.n_class**2).reshape(self.n_class, self.n_class)

    def get_eval_results(self):
        """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
        hist = self.hist
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        #cls_iu = dict(zip(range(self.n_class), iu))
        cls_iu = iu
        return {'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu,}, cls_iu

#def getConfMatrixResults(matrix):
#    assert(len(matrix.shape)==2 and matrix.shape[0]==matrix.shape[1])
#
#    count_correct = np.diag(matrix)
#    count_preds   = matrix.sum(1)
#    count_gts     = matrix.sum(0)
#    epsilon       = np.finfo(np.float32).eps
#    accuracies    = count_correct / (count_gts + epsilon)
#    IoUs          = count_correct / (count_gts + count_preds - count_correct + epsilon)
#    totAccuracy   = count_correct.sum() / (matrix.sum() + epsilon)
#
#    num_valid     = (count_gts > 0).sum()
#    meanAccuracy  = accuracies.sum() / (num_valid + epsilon)
#    meanIoU       = IoUs.sum() / (num_valid + epsilon)
#
#    return {'totAccuracy': round(totAccuracy,4), 'meanAccuracy': round(meanAccuracy,4), 'meanIoU': round(meanIoU,4)}
