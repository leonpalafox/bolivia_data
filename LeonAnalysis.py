#Analysis of data with pandas
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
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


#create color index

color_index = database.MaritalStatus.values
color_index = pd.get_dummies(database.BCountry).values.argmax(1) #for countries
#make dict index to country
name_to_index = pd.get_dummies(database.BCountry).keys()
NUM_COLORS = len(np.unique(color_index))
cm = plt.cm.get_cmap('gist_rainbow')
 #analyze red variables
 #index 28 - > 36      
sub_data = database.iloc[:,28:37]
data_matrix = sub_data.values
model = TSNE(n_components=3, random_state=0)

projected =  model.fit_transform(data_matrix) 
for idx in np.unique(color_index):
    plt.scatter(projected[color_index == idx, 0], projected[color_index == idx, 1], projected[color_index == idx, 2], c = cm(1.*idx/NUM_COLORS), label = name_to_index[idx])
plt.legend()