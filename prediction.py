import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def train_test(df):

    
    train = df.iloc[60000:108952]
    test =  df.iloc[108953:]
    
    return train, test



def rf_modelisation(x_train, y_train):
    ''' determination des hyperparametre de RF'''
    weights = np.linspace(0.1,0.9,100)
    params = [{
        "n_estimators": [10, 100,150],
        "max_features": [2, 4, 8,10,12],
        "class_weight":[{0:x, 1:1.0-x} for x in weights]
        }]

    rfCV = GridSearchCV(
        RandomForestClassifier(),
        params,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        return_train_score=True)
    
    rfCV = rfCV.fit(x_train, y_train)
    
    return rfCV.best_estimator_

#
def logist_modelisation(x_train, y_train):
    # grille de valeurs
    weights = np.linspace(0.1,0.9,100)

    params = [{"C": [0.01, 0.2, 0.5, 1, 5, 10, 20],
           #"penalty": ["l1", "l2"],
           "class_weight":[{0:x, 1:1.0-x} for x in weights]
          }]

    logitCV = GridSearchCV(
        LogisticRegression(),
        params,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        return_train_score=True)
    
    logitCV = logitCV.fit(x_train, y_train)
    
    return logitCV.best_estimator_

#
def OneSVM_modelisation(x_train, y_train):
    # grille de valeurs

    params = [{"nu": [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9], 
              'max_iter': [100,250, 500, 700, 900],
           #"class_weight":[{0:x, 1:1.0-x} for x in weights]
          }]

    outlierCV = GridSearchCV(
        OneClassSVM(),
        params,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        return_train_score=True)
    
    outlierCV = outlierCV.fit(x_train, y_train)
    
    
    return outlierCV.best_estimator_

def DT_modelisation(x_train, y_train):
    # grille de valeurs
    weights = np.linspace(0.1,0.9,100)

    params = [{
        "max_depth": [3, 5, 10, 15,None],
        "min_samples_split": [2, 5, 10,15,20,30],
        "min_samples_leaf": [1, 2, 5,10,30,50],
        "class_weight":[{0:x, 1:1.0-x} for x in weights]
        }]

    dtCV = GridSearchCV(
        DecisionTreeClassifier(),
        params,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        return_train_score=True)
    
    dtCV = dtCV.fit(x_train, y_train)
    
    return dtCV.best_estimator_
  
# Chargement des données
data = pd.read_csv("data/all_features.csv", sep=';',low_memory=False)
label = pd.read_csv("data/descriptif_hiver_ete.csv", sep=';',low_memory=False)
print(len(data))


# definition de train et test 
# definition de x et y 
x_train, x_test = train_test(data.iloc[:,1:])
train_target,test_target = train_test(label)
y_test = test_target['baignade']
y_train = train_target['baignade']

print(len(x_train))
print(len(x_test))
x_test
## Logistic Regression
model_logist = logist_modelisation(x_train, y_train)
model_logist
## decision tree
model_dt = DT_modelisation(x_train, y_train)
model_dt



## Random forest
model_rf = rf_modelisation(x_train, y_train)
model_rf


import joblib
joblib.dump(model_dt, "saved_models/decision_tree2.joblib")
joblib.dump(model_rf, "saved_models/random_forest2.joblib")
joblib.dump(model_logist, "saved_models/reg_logist2.joblib")
# Evaluation


## RF
model_rf.fit(x_train, y_train)

# prediction
y_train_predict = model_rf.predict(x_train)
y_test_predict = model_rf.predict(x_test)

# Evaluation
print("-----------------------Training data-----------------------")
print(classification_report(y_train, y_train_predict))
print("-------------------------Test data-------------------------")
print(classification_report(y_test, y_test_predict))
model_logist.fit(x_train, y_train)

# prediction
y_train_predict = model_logist.predict(x_train)
y_test_predict = model_logist.predict(x_test)

# Evaluation
print("-----------------------Training data-----------------------")
print(classification_report(y_train, y_train_predict))
print("-------------------------Test data-------------------------")
print(classification_report(y_test, y_test_predict))
model_logist.fit(x_train, y_train)

# prediction
y_train_predict = model_logist.predict(x_train)
y_test_predict = model_logist.predict(x_test)

# Evaluation
print("-----------------------Training data-----------------------")
print(classification_report(y_train, y_train_predict))
print("-------------------------Test data-------------------------")
print(classification_report(y_test, y_test_predict))
model_dt.fit(x_train, y_train)

# prediction
y_train_predict = model_dt.predict(x_train)
y_test_predict = model_dt.predict(x_test)

# Evaluation
print("-----------------------Training data-----------------------")
print(classification_report(y_train, y_train_predict))
print("-------------------------Test data-------------------------")
print(classification_report(y_test, y_test_predict))
model_dt.fit(x_train, y_train)

# prediction
y_train_predict = model_dt.predict(x_train)
y_test_predict = model_dt.predict(x_test)

# Evaluation
print("-----------------------Training data-----------------------")
print(classification_report(y_train, y_train_predict))
print("-------------------------Test data-------------------------")
print(classification_report(y_test, y_test_predict))

df = x_test
df['true_label'] = y_test
df['predicted_label'] = y_test_predict
df
from sklearn import metrics
def roc_curves_plot(y_train, pred_proba_train, y_test, pred_proba_test):
    false_positive_rate_train, true_positive_rate_train, _ = metrics.roc_curve(
        y_train, pred_proba_train)
    roc_auc_train = metrics.auc(false_positive_rate_train,
                                true_positive_rate_train)

    false_positive_rate_test, true_positive_rate_test, _ = metrics.roc_curve(
        y_test, pred_proba_test)
    roc_auc_test = metrics.auc(false_positive_rate_test,
                               true_positive_rate_test)

    plt.title('Receiver Operating Characteristic')
    plt.plot(
        false_positive_rate_train,
        true_positive_rate_train,
        'b',
        label='AUC Train = %0.4f' % roc_auc_train)
    plt.plot(
        false_positive_rate_test,
        true_positive_rate_test,
        'g',
        label='AUC Test = %0.4f' % roc_auc_test)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    


# ********************************* dataviz



def prediction(model , data_x):
    
    df = pd.DataFrame()
    
    pred_proba = model.predict_proba(data_x)
    proba_class_1 = list(pred_proba[:,1])
    
    df['probability'] = proba_class_1
    df['country'] = np.nan
    
    return df

