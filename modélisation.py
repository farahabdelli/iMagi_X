
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplt
import plotly as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# %matplotlib inline
pd.options.mode.chained_assignment = None

def train_test(df):

    
    train = df.iloc[75000:106952]
    test =  df.iloc[106952:]
    
    return train, test

"""# modeling"""

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
        cv=5,
        n_jobs=-1,
        return_train_score=True)
    
    rfCV = rfCV.fit(x_train, y_train)
    
    return rfCV.best_estimator_

#
def logist_modelisation(x_train, y_train):
    # grille de valeurs
    weights = np.linspace(0.1,0.9,100)

    params = [{"C": [0.01, 0.2, 0.5, 1, 5, 10, 20],
           "penalty": [ "l2"],
           "max_iter": [ 5000],
           "class_weight":[{0:x, 1:1.0-x} for x in weights]
          }]

    logitCV = GridSearchCV(
        LogisticRegression(),
        params,
        scoring="recall",
        cv=5,
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
        cv=5,
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
        cv=5,
        n_jobs=-1,
        return_train_score=True)
    
    dtCV = dtCV.fit(x_train, y_train)
    
    return dtCV.best_estimator_

# load data
data = pd.read_csv("data/all_features.csv", sep=';',low_memory=False)
label = pd.read_csv("data/descriptif_hiver_ete.csv", sep=';',low_memory=False)
print(len(data))


# train  test partition
x_train, x_test = train_test(data.iloc[:,1:])
train_target,test_target = train_test(label)
y_test = test_target['baignade']
y_train = train_target['baignade']

## Logistic Regression

model_logist = logist_modelisation(x_train, y_train)
model_logist

## decision tree
model_dt = DT_modelisation(x_train, y_train)
model_dt

## Random forest
model_rf = rf_modelisation(x_train, y_train)
model_rf

#save model
import joblib
joblib.dump(model_dt, "saved_models/decision_tree2.joblib")
joblib.dump(model_rf, "saved_models/random_forest2.joblib")
joblib.dump(model_logist, "saved_models/reg_logist2.joblib")

## fit RF model
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

model_dt.fit(x_train, y_train)

# prediction
y_train_predict = model_dt.predict(x_train)
y_test_predict = model_dt.predict(x_test)

# Evaluation
print("-----------------------Training data-----------------------")
print(classification_report(y_train, y_train_predict))
print("-------------------------Test data-------------------------")
print(classification_report(y_test, y_test_predict))

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

roc_curves_plot(y_train, y_train_predict, y_test, y_test_predict)

roc_curves_plot(y_train, y_train_predict, y_test, y_test_predict)

accur_train = accuracy_score(y_train, y_train_predict)
precis_train = precision_score(y_train, y_train_predict, average='micro')
rappel_train = recall_score(y_train, y_train_predict, average='micro')
F1_train = f1_score(y_train, y_train_predict, average='micro')

# metrics on test
accur_test = accuracy_score(y_test, y_test_predict)
precis_test = precision_score(y_test, y_test_predict)
rappel_test = recall_score(y_test, y_test_predict)
F1_test = f1_score(y_test, y_test_predict)


print(accur_test )
print(precis_test )
print(rappel_train )
print(F1_test )

"""# under sampling"""

# Chargement des données
data = pd.read_csv("data/all_features.csv", sep=';',low_memory=False)
label = pd.read_csv("data/descriptif_hiver_ete.csv", sep=';',low_memory=False)
print(len(data))
data = data.set_index('date')
label = label.set_index('date')

all= pd.merge(data, label, how="left", on=["date"])
all = all.reset_index()
all

X =all.drop(["baignade"],axis=1).iloc[60000:]
y = all["baignade"][60000:]

import imblearn
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler 
rs = RandomUnderSampler(random_state=42)

#print('Original dataset shape %s' % Counter(y))

X_res, y_res = rs.fit_resample(X, y)

print('After undersample dataset shape %s' % Counter(y_res))

Y = pd.DataFrame(y_res,columns=["baignade"])
all_res= pd.concat([X_res, Y],axis=1)
all_res= all_res.sort_values('date').reset_index()
all_res = all_res.drop(['index'],axis=1)
all_res

def train_test(df):

    
    train = df.iloc[:1500,1:]
    test =  df.iloc[1501:,1:]
    
    return train, test

# definition de train et test 
# definition de x et y 
train, test = train_test(all_res)

explicative_columns = [x for x in train.columns if x not in "baignade"]
y_train = train.baignade
y_train = pd.DataFrame(y_train,columns=["baignade"])
x_train = train[explicative_columns]

y_test = test.baignade
y_test = pd.DataFrame(y_test,columns=["baignade"])
x_test = test[explicative_columns]

# class partition before and after undersampling

import plotly.express as px
fig=px.histogram(y_train,y_train.columns[0],color=y_train.columns[0], width=800, height=400)

fig.show()

#model pipelines

def rf_modelisation(x_train, y_train):
    weights = np.linspace(0.3,0.7,100)
    scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score' : make_scorer(f1_score)
}
    params = [{
        "n_estimators": [10, 100,150],
        "max_features": [2, 4, 8,10,12],
        "class_weight":[{0:x, 1:1.0-x} for x in weights]
        }]

    rfCV = GridSearchCV(
        RandomForestClassifier(),
        params,
        scoring=scorers,
        cv=5,
        n_jobs=-1,
        refit='recall_score',
        return_train_score=True)
    
    rfCV = rfCV.fit(x_train, y_train)
    
    return rfCV.best_estimator_

#
def logist_modelisation(x_train, y_train):
    # grille de valeurs
    weights = np.linspace(0.3,0.7,100)
    scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score' : make_scorer(f1_score)}
    params = [{"C": [0.01, 0.2, 0.5, 1, 5, 10, 20],
           "penalty": [ "l2"],
           "max_iter": [ 5000],
           "class_weight":[{0:x, 1:1.0-x} for x in weights]
          }]

    logitCV = GridSearchCV(
        LogisticRegression(),
        params,
        scoring=scorers,
        cv=5,
        n_jobs=-1,
        refit='recall_score',
        return_train_score=True)
    
    logitCV = logitCV.fit(x_train, y_train)
    
    return logitCV.best_estimator_
    
    
def DT_modelisation(x_train, y_train):
    # grille de valeurs
    weights = np.linspace(0.3,0.7,100)
    scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score' : make_scorer(f1_score)
}
    params = [{
        "max_depth": [3, 5, 10, 15,None],
        "min_samples_split": [2, 5, 10,15,20,30],
        "min_samples_leaf": [1, 2, 5,10,30,50],
        "class_weight":[{0:x, 1:1.0-x} for x in weights]
        }]

    dtCV = GridSearchCV(
        DecisionTreeClassifier(),
        params,
        scoring=scorers,
        cv=5,
        n_jobs=-1,
        refit='recall_score',
        return_train_score=True)
    
    dtCV = dtCV.fit(x_train, y_train)
    
    return dtCV.best_estimator_

## Logistic Regression

model_logist = logist_modelisation(x_train, y_train)
model_logist

model_logist.fit(x_train, y_train)

# prediction
y_train_predict = model_logist.predict(x_train)
y_test_predict = model_logist.predict(x_test)

# Evaluation
print("-----------------------Training data-----------------------")
print(classification_report(y_train, y_train_predict))
print("-------------------------Test data-------------------------")
print(classification_report(y_test, y_test_predict))

## decision tree
model_dt = DT_modelisation(x_train, y_train)
model_dt

model_dt.fit(x_train, y_train)

# prediction
y_train_predict = model_dt.predict(x_train)
y_test_predict = model_dt.predict(x_test)

# Evaluation
print("-----------------------Training data-----------------------")
print(classification_report(y_train, y_train_predict))
print("-------------------------Test data-------------------------")
print(classification_report(y_test, y_test_predict))

## Random forest
model_rf = rf_modelisation(x_train, y_train)
model_rf

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

import joblib
joblib.dump(model_dt, "saved_models/decision_tree4.joblib")
joblib.dump(model_rf, "saved_models/random_forest4.joblib")
joblib.dump(model_logist, "saved_models/reg_logist4.joblib")

"""# over & under sampling SMOTE-ENN"""

# Chargement des données
data = pd.read_csv("data/all_features.csv", sep=';',low_memory=False)
label = pd.read_csv("data/descriptif_hiver_ete.csv", sep=';',low_memory=False)
print(len(data))
data = data.set_index('date')
label = label.set_index('date')


all= pd.merge(data, label, how="left", on=["date"])
all = all.reset_index()
all
X =all.drop(["baignade","date"],axis=1).iloc[65000:]
y = all["baignade"][65000:]
import imblearn
from collections import Counter
from imblearn.combine import SMOTEENN
SMOTE = SMOTEENN()

X_res, y_res = SMOTE.fit_resample(X, y)

print('After undersample dataset shape %s' % Counter(y_res))

import plotly.express as px
fig=px.histogram(label,label.columns[0],color=label.columns[0], width=800, height=400)

fig.show()

Y = pd.DataFrame(y_res,columns=["baignade"])
all_res= pd.concat([X_res, Y],axis=1)

all_res = all_res.sample(frac=1).reset_index(drop=True)
all_res

def train_test(df):

    
    train = df.iloc[:58000]
    test =  df.iloc[58001:]
    
    return train, test
# definition de train et test 
# definition de x et y 
train, test = train_test(all_res)

explicative_columns = [x for x in train.columns if x not in "baignade"]
y_train = train.baignade
y_train = pd.DataFrame(y_train,columns=["baignade"])
x_train = train[explicative_columns]

y_test = test.baignade
y_test = pd.DataFrame(y_test,columns=["baignade"])
x_test = test[explicative_columns]

import plotly.express as px
fig=px.histogram(y_test,y_test.columns[0],color=y_test.columns[0], width=800, height=400)

fig.show()

def rf_modelisation(x_train, y_train):
    weights = np.linspace(0.3,0.7,100)
    scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score' : make_scorer(f1_score)
}
    params = [{
        "n_estimators": [10, 100,150],
        "max_features": [2, 4, 8,10,12],
        "class_weight":[{0:x, 1:1.0-x} for x in weights]
        }]

    rfCV = GridSearchCV(
        RandomForestClassifier(),
        params,
        scoring=scorers,
        cv=5,
        n_jobs=-1,
        refit='recall_score',
        return_train_score=True)
    
    rfCV = rfCV.fit(x_train, y_train)
    
    return rfCV.best_estimator_

#
def logist_modelisation(x_train, y_train):
    # grille de valeurs
    weights = np.linspace(0.3,0.7,100)
    scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score' : make_scorer(f1_score)}
    params = [{"C": [0.01, 0.2, 0.5, 1, 5, 10, 20],
           "penalty": [ "l2"],
           "max_iter": [ 5000],
           "class_weight":[{0:x, 1:1.0-x} for x in weights]
          }]

    logitCV = GridSearchCV(
        LogisticRegression(),
        params,
        scoring=scorers,
        cv=5,
        n_jobs=-1,
        refit='recall_score',
        return_train_score=True)
    
    logitCV = logitCV.fit(x_train, y_train)
    
    return logitCV.best_estimator_
    
    
def DT_modelisation(x_train, y_train):
    # grille de valeurs
    weights = np.linspace(0.3,0.7,100)
    scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score' : make_scorer(f1_score)
}
    params = [{
        "max_depth": [3, 5, 10, 15,None],
        "min_samples_split": [2, 5, 10,15,20,30],
        "min_samples_leaf": [1, 2, 5,10,30,50],
        "class_weight":[{0:x, 1:1.0-x} for x in weights]
        }]

    dtCV = GridSearchCV(
        DecisionTreeClassifier(),
        params,
        scoring=scorers,
        cv=5,
        n_jobs=-1,
        refit='recall_score',
        return_train_score=True)
    
    dtCV = dtCV.fit(x_train, y_train)
    
    return dtCV.best_estimator_

## Logistic Regression

model_logist = logist_modelisation(x_train, y_train)
model_logist

## decision tree
model_dt = DT_modelisation(x_train, y_train)
model_dt

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

all_res.to_csv("data/all_features_res.csv",sep=";",index=False)

def train_test(df):

    train = df.iloc[:58000]
    test =  df.iloc[58001:]
    
    return train, test

# definition de train et test 

x_train, x_test = train_test(data)#sélectionner que les features sans la colonne date

y_train, y_test = train_test(label)#sélectionner que la colonne cible "baignade"

all.to_csv("data/all_features.csv",sep=";",index=False)

