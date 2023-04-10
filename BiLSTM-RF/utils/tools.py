# -*- coding: utf-8 -*-

import itertools
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pandas as pd
import math


def read_result_file(filename='result/result_file'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        # protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            if index % 3 == 0:
                rna = values[0]
            if index % 3 != 0:
                results.setdefault(rna, []).append(values)

            index = index + 1

    return results


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    if (tp + fn) == 0:
        q9 = float(tn - fp) / (tn + fp + 1e-06)
    if (tn + fp) == 0:
        q9 = float(tp - fn) / (tp + fn + 1e-06)
    if (tp + fn) != 0 and (tn + fp) != 0:
        q9 = 1 - float(np.sqrt(2)) * np.sqrt(
            float(fn * fn) / ((tp + fn) * (tp + fn)) + float(fp * fp) / ((tn + fp) * (tn + fp)))

    Q9 = (float)(1 + q9) / 2
    accuracy = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    recall = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn) / (tn + fp + 1e-06)
    ppv = float(tp) / (tp + fp + 1e-06)
    npv = float(tn) / (tn + fn + 1e-06)
    f1_score = float(2 * tp) / (2 * tp + fp + fn + 1e-06)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, f1_score, Q9, ppv, npv

# def draw_roc(y_test, y_score):
#     # Compute ROC curve and ROC area for each class
#     n_classes=y_score.shape[-1]
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     num=0
#     if n_classes<=1:
#         fpr[0], tpr[0], _ = roc_curve(y_test[:,], y_score[:,])
#         roc_auc[0] = auc(fpr[0], tpr[0])
#         num=0
#     else:    
#         for i in range(n_classes):            
#             fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#             roc_auc[i] = auc(fpr[i], tpr[i])
#             num=n_classes-1

#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#     plt.figure(figsize=(10, 10))

#     #line-width
#     lw = 2
#     auc_score=roc_auc[num]*100
#     plt.plot(fpr[num], tpr[num], color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f%%)' % auc_score)
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.05])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()

# def draw_pr(y_test, y_score):    
#     # Compute ROC curve and ROC area for each class
#     n_classes=y_score.shape[-1]
#     precision = dict()
#     recall = dict()
#     average_precision = dict()
#     num=0
#     if n_classes<=1:        
#         precision[0], recall[0], _ = precision_recall_curve(y_test[:, ],y_score[:,])
#         average_precision[0] = average_precision_score(y_test[:, ], y_score[:, ])
#         num=0
#     else:    
#         for i in range(n_classes):           
#             precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
#             average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
#             num=n_classes-1

#     # Compute micro-average ROC curve and ROC area
#     precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
#     average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")

#     # Plot Precision-Recall curve
#     plt.figure(figsize=(10, 10))

#     #line-width
#     lw = 2
#     pr_score=average_precision[num]*100
#     plt.plot(recall[i], precision[i], color='darkorange', lw=lw,
#              label='Precision-recall curve (area = %0.2f%%)' % pr_score)
#     plt.xlim([0.0, 1.05])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall curve')
#     plt.legend(loc="lower right")
#     plt.show()

# def plot_embedding(tr_X_AC, y,title=None):
#     #将数据归一化到0-1之间
#     x_min, x_max = np.min(tr_X_AC, 0), np.max(tr_X_AC, 0)
#     tr_X_AC = (tr_X_AC - x_min) / (x_max - x_min)

#     df = pd.DataFrame(dict(x=tr_X_AC[:,0],y=tr_X_AC[:,1], label=y))
#     groups = df.groupby('label')

#     plt.figure(figsize=(10, 10))
#     plt.subplot(111)
#     for name, group in groups:
#         plt.scatter(group.x, group.y,c=plt.cm.Set1(name / 10.),label=name)
#         #    plt.text(tr_X_AC[i, 0], tr_X_AC[i, 1], '.',
#         #         color=plt.cm.Set1(labels[i] / 10.),
#         #         fontdict={'weight': 'bold', 'size': 10})
#     plt.xticks([]), plt.yticks([])
#     plt.legend()
#     if title is not None:
#         plt.title(title)
#     plt.show()
