# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:27:08 2021

@author: owner

pyopls - Orthogonal Projection to Latent Structures in Python.
code link: https://github.com/BiRG/pyopls#opls-and-pls-da
OPLS and PLS-DA example
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import roc_curve, roc_auc_score
from pyopls import OPLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder 
import os
os.chdir('D:/Canada/University/PhD/Research/Programs/Python/MV/codes/Visualization')


# spectra = pd.read_csv('data/DATA_ALL-TRAIN-TEST.csv', index_col=0)
data = pd.read_csv('data/DATA_ALL-TRAIN-TEST.csv')
labels = "type"
features_to_exclude = ["index", "Img_num", "cont.num", labels]
print(data.head)
heads = list(data.columns)
x_train = data
# test = spectra[spectra.classification.isin(['beadpack','limestone', 'sandstone'])]
types = pd.Categorical(data[labels])
types = types.categories
data = data[data[labels].isin([types[0], types[1], types[2]])]
# spectra[labels] = spectra["classification"].astype('category')
# spectra.dtypes
print(data[labels].value_counts())
# data[labels] = LabelEncoder().fit_transform(data[labels])

# Plot features histogram
data[0:800].drop(features_to_exclude, axis=1).hist(figsize = (24, 20), color='r', alpha=0.9)
data[800:1600].drop(features_to_exclude, axis=1).hist(figsize = (24, 20), color='g', alpha=0.9)
data[1600:2400].drop(features_to_exclude, axis=1).hist(figsize = (24, 20), color='b', alpha=0.9)
data.drop(features_to_exclude, axis=1).hist(figsize = (24, 20), alpha=0.85)
plt.show()


# Plot Heat Map represtening the correlation between the features
data_red = data.drop(["index", "Img_num", "cont.num"], axis=1)
data_red = data_red.rename(columns={"type":"Rock Type", 
                                    "porosity":"Porosity",
                                    "specific.surf.area":"SSA", 
                                    "convexity":"Convexity",
                                    "area(m00)":"Area(m00)"})
C_mat = data_red.corr()
fig = plt.figure(figsize = (10, 10))
plt.title ("Rock Features Heat Map\n", fontsize=14)
sb.heatmap(C_mat, cmap ="RdBu", alpha=0.85, square = True, linewidths=0.01) #RdBu, coolwarm
plt.savefig("Rock_Features_Heat_Map"+'.png', dpi=300)
plt.show()



# data[labels] = LabelEncoder().fit_transform(data[labels])
for i in range(len(types)):
    data = data.replace(types[i], i)



y_train = data[labels]
x_train = data.drop(features_to_exclude, axis=1)


opls = OPLS(1)
x_opls = opls.fit_transform(x_train, y_train)
r2_x_opls = opls.score(x_train)

pls = PLSRegression(n_components=7)
x_pls = pls.fit_transform(x_train, y_train)[0]
r2_x_pls = pls.score(x_train, y_train)


alpha = 1
with plt.style.context(('ggplot')):
    plt.figure(1)
    plt.scatter(x_pls[0:800, 0], x_pls[0:800, 1], c='r', edgecolors='k', alpha=alpha, label=types[0])
    plt.scatter(x_pls[800:1600, 0], x_pls[800:1600, 1], c='g', edgecolors='k', alpha=alpha, label=types[1])
    plt.scatter(x_pls[1600:2400, 0], x_pls[1600:2400, 1], c='b', edgecolors='k', alpha=alpha, label=types[2])
    x_limit = max(abs(x_pls[:,0]))
    y_limit = max(abs(x_pls[:,1]))
    factor = 1.1
    plt.xlim(x_limit*-1*factor, x_limit*factor)
    plt.ylim(y_limit*-1*factor, y_limit*factor*1.2)
    plt.title('PLS Scores')
    plt.xlabel('t1')
    plt.ylabel('t2')
    plt.legend(loc='upper right')
    plt.show()

with plt.style.context(('ggplot')):
    plt.figure(2)
    plt.scatter(x_opls[0:800, 0], x_opls[0:800, 1], c='r',edgecolors='k', alpha=alpha, label=types[0])
    plt.scatter(x_opls[800:1600, 0], x_opls[800:1600, 1], c='g',edgecolors='k', alpha=alpha, label=types[1])
    plt.scatter(x_opls[1600:2400, 0], x_opls[1600:2400, 1], c='b',edgecolors='k', alpha=alpha, label=types[2])
    x_limit = max(abs(x_opls[:,0]))
    y_limit = max(abs(x_opls[:,1]))
    factor = 1.1
    plt.xlim(x_limit*-1*factor, x_limit*factor)
    plt.ylim(y_limit*-1*factor, y_limit*factor*1.2)
    plt.title('OPLS Scores')
    plt.xlabel('t1')
    plt.ylabel('t2')
    plt.legend(loc='upper right')
    plt.show()


def pls_da(X_train, y_train, X_test):
    
    # Define the PLS object for binary classification
    plsda = PLSRegression(n_components=2)
    
    # Fit the training set
    plsda.fit(X_train, y_train)
    
    # Binary prediction on the test set, done with thresholding
    binary_prediction = (pls_binary.predict(X_test)[:,0] > 0.5).astype('uint8')
    
    return binary_prediction




