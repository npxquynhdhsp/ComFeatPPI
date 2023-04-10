import pickle

import numpy as np
from utils.report_result import print_metrics, my_cv_report

result = pickle.load(open("SAE_eval_Human.pkl", "rb"))
print(result.keys())

cv_scores = []
for i in range(5):
    try:
        fold_i = "subset" + str(i)
        # print(result[fold_i])
        prob_Y = result[fold_i].get("prob_y")
        true_y = result[fold_i].get("true_y")
    except KeyError:
        fold_i = "fold" + str(i)
        # print(result[fold_i])
        prob_Y = result[fold_i].get("prob_y")
        if prob_Y is None:
            prob_Y = result[fold_i].get("prob")
        true_y = result[fold_i].get("true_y")
        if true_y is None:
            true_y = result[fold_i].get("true")

    if len(true_y.shape) > 1:
        true_y = np.argmax(true_y, axis=1)
    cv_scores.append(print_metrics(true_y, prob_Y, verbose=1))

print('\n--- Final')
my_cv_report(cv_scores)
