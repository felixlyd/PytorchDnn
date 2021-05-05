import os

import numpy as np
import torch
from sklearn import metrics


def get_test_acc_loss(model_, data_iter, criterion_, test=False, out_path=None):
    model_.eval()
    total_loss = 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for inputs, labels in data_iter:
            outputs = model_(inputs)
            loss = criterion_.compute_(outputs, labels)
            total_loss = total_loss + loss
            true = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, true)
            predict_all = np.append(predict_all, predict)
        acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        print("Precision, Recall and F1-Score...")
        print(report)
        print("Confusion Matrix...")
        print(confusion)
        if out_path is not None:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            result_path = os.path.join(out_path, "result.txt")
            np.savetxt(result_path, np.array([labels_all, predict_all]).T, fmt='%d')
            report_path = os.path.join(out_path, "report.txt")
            with open(report_path, 'w') as wf:
                wf.write(report)
            confusion_path = os.path.join(out_path, "confusion.txt")
            np.savetxt(confusion_path, confusion, fmt="%d")
    return acc, total_loss / len(data_iter)


def get_train_acc(outputs, labels):
    acc_nums = 0
    predict = torch.max(outputs.data, 1)[1]
    true = labels.data
    acc_nums = acc_nums + torch.sum(true == predict)
    return acc_nums / len(outputs)
