#Analysis of data with pandas
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import pylab as plt
database = pd.read_csv('base.csv')
rows, col = database.shape
#This step cleans the data
for keyIdx in database.keys():
    temp_ = database[keyIdx].isnull().as_matrix()
    num_nan = np.sum(temp_.astype(int))
    perc = float(num_nan)/rows
    if perc >= 0.2:
        del database[keyIdx]
    else:
        database = database[pd.notnull(database[keyIdx])]

database.BCountry = database.BCountry.replace({'boliva':'Bolivia'})
database.BCountry = database.BCountry.replace({'Boliva':'Bolivia'})
database.BCountry = database.BCountry.replace({'Bolvia':'Bolivia'})
database.BCountry = database.BCountry.replace({'Brasil':'Brazil'})

#depurate base for countries
country_counts = database.BCountry.value_counts()
for country in database.BCountry.unique():
    if country_counts[country] <100:
        database = database[database.BCountry != country]

rows, col = database.shape
database = database.reset_index(drop = True)
class_label = database.HouseholdSize > 2
class_label = class_label.astype(int)


 #analyze red variables
 #index 28 - > 36      
sub_data = database.iloc[:,28:37]
data_matrix = sub_data.values
X_train, X_test, y_train, y_test = train_test_split(data_matrix, class_label.values, test_size=0.33, random_state=42)
regr = linear_model.ElasticNetCV(l1_ratio = np.linspace(0.001,1,300))
#extract classes
clf_l1_LR = linear_model.LogisticRegression(C=100, penalty='l1', tol=0.01)
clf_l2_LR = linear_model.LogisticRegression(C=100, penalty='l2', tol=0.01)
clf_l1_LR.fit(X_train, y_train)
clf_l2_LR.fit(X_train, y_train)
print clf_l1_LR.score(X_test, y_test)
print clf_l2_LR.score(X_test, y_test)
