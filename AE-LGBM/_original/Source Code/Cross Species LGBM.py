
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix 


#os.chdir('C:/Protein')

# load or create your dataset
Mydata = pd.read_csv("gbm_human.csv", header=0)
Mydata.columns.name = None

data = Mydata.drop(Mydata.columns[Mydata.shape[1]-1], axis=1)
label = Mydata.iloc[:,(Mydata.shape[1]-1)]


data_tr=data
label_tr=label

os.chdir('C:/Protein/Final Revision/independent species')
test=pd.read_csv("gbm_ecoli.csv", header=0)
test.columns.name = None

data_te= test.drop(test.columns[test.shape[1]-1], axis=1)
label_te= test.iloc[:,(test.shape[1]-1)]

# create dataset for lightgbm
lgb_train = lgb.Dataset(data_tr, label_tr)
lgb_eval = lgb.Dataset(data_te, label_te, reference=lgb_train)

# specify your configurations as a dict
params = {
  'boosting_type': 'gbdt',
  'objective': 'binary',
  'metric': 'binary_logloss',
  'num_leaves': 80, #50 for Yeast
  'learning_rate': 0.05,
  'feature_fraction': 0.9,
  'bagging_fraction': 0.8,
  'bagging_freq': 5,
  'verbose': 0
}


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_eval,
                early_stopping_rounds=100)

#gbm.save_model(["model"+str(k)+".txt"][0])

y_pred = gbm.predict(data_te, num_iteration=gbm.best_iteration)
for i in range(0,len(list(y_pred))):
  
    if y_pred[i]>=0.5:  
        y_pred[i]=1
    else:  
        y_pred[i]=-1

label_te.reset_index(drop=True, inplace=True)
pd.DataFrame(list(y_pred)).reset_index(drop=True, inplace=True)

#Confusion matrix
cm = confusion_matrix(label_te, y_pred)
pd.DataFrame(cm)
Tp=cm[1,1]
Fp=cm[0,1] 
Tn=cm[0,0] 
Fn=cm[1,0]
Sn=Tp/(Tp+Fn)
Sp=Tn/(Tn+Fp)
Acc=(Tp+Tn)/np.sum(cm)
Mcc= (Tp*Tn - Fp*Fn)/((((Tp+Fp)*(Tp+Fn))**(1/2))*(((Tn+Fp)*(Tn+Fn))**(1/2)))







