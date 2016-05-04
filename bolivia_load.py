import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.manifold import TSNE
import pylab as plt
my_data = np.genfromtxt('DataBaseBolivia.csv', delimiter=',',dtype="|S5")
        