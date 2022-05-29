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
data[labels] = LabelEncoder().fit_transform(data[labels])

"""
# Plot features histogram
data[0:800].drop(features_to_exclude, axis=1).hist(figsize = (24, 20), color='r', alpha=0.9)
data[800:1600].drop(features_to_exclude, axis=1).hist(figsize = (24, 20), color='g', alpha=0.9)
data[1600:2400].drop(features_to_exclude, axis=1).hist(figsize = (24, 20), color='b', alpha=0.9)
data.drop(features_to_exclude, axis=1).hist(figsize = (24, 20), alpha=0.85)
plt.show()
"""

# Plot Heat Map represtening the correlation between the features
data_red = data.drop(["index", "Img_num", "cont.num"], axis=1)
data_red = data_red.rename(columns={"type":"Rock Type", 
                                    "porosity":"Porosity",
                                    "specific.surf.area":"P/A", 
                                    "convexity":"Convexity",
                                    "area(m00)":"Area(m00)"})

C_mat = data_red.corr()
fig = plt.figure(figsize = (12, 10))
# plt.title ("Rock Features Heat Map\n", fontsize=16)
sb.heatmap(C_mat, cmap ="RdBu", vmin=-1, vmax=1, square = True, annot=False, annot_kws={"size":12}, linewidths=0, alpha=0.8) #RdBu, coolwarm
# sb.set(font_scale=1.5) # font size 2
# dpi - sets the resolution of the saved image in dots/inches
# bbox_inches - when set to 'tight' - does not allow the labels to be cropped
plt.savefig("Rock_Features_Heat_Map"+'.png', dpi=200, bbox_inches='tight')
plt.show()





