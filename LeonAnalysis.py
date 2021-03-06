#Analysis of data with pandas
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import pylab as plt
import sys
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
class_label = database.RTime > 20
class_label = class_label.astype(int)
database['Classes'] = pd.Series(class_label)
#create color index
variable_to_plot = 'BCountry'
color_index = pd.get_dummies(database[variable_to_plot]).values.argmax(1) #for countries
#make dict index to country
name_to_index = pd.get_dummies(database[variable_to_plot]).keys()
NUM_COLORS = len(np.unique(color_index))
cm = plt.cm.get_cmap('gist_rainbow')
sys.exit()
 
 #analyze red variables
 #index 28 - > 36      
sub_data = database.iloc[:,28:37]
sub_data = database.iloc[:,37:67]
data_matrix = sub_data.values
model = TSNE(n_components=2, random_state=0)
fig = plt.figure()
ax = fig.add_subplot(111)
projected =  model.fit_transform(data_matrix) 
for idx in np.unique(color_index):
    ax.scatter(projected[color_index == idx, 0], projected[color_index == idx, 1], c = cm(1.*idx/NUM_COLORS), label = name_to_index[idx])
plt.legend()