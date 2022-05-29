# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:37:27 2021

@author: owner
"""


from numpy.random import seed
seed(1)
import numpy as np
from numpy import genfromtxt
import pandas as pd
import json
import os

from contextlib import redirect_stdout
# Visualization for results
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.chdir('D:/Canada/University/PhD/Research/Programs/Python/MV/codes/Visualization')


model = "PCA"
x_label = 't1'
y_label =  't2'
data_file = model+' scores '+x_label+'-'+y_label

X = 1
Y = 1
if model == 'OPLS':
    X = 1.00088
    Y = 1.02916
elif model == 'PLS':
    xlim = [-20, 10]
    ylim = [-12, 10]
elif model == 'PCA':
    xlim = [-30, 10]
    ylim = [-10, 15]
elif model == 'SIMCA':
    xlim = [-10, 30]
    ylim = [-25, 30]
    
if model == 'OPLS' or model == 'PLS' or model == 'PCA':
    data = genfromtxt("data/"+data_file+".csv",delimiter=',')
    bp_x = data[2:, 0]/X
    bp_y = data[2:, 1]/Y
    ss_x = data[2:, 2]/X
    ss_y = data[2:, 3]/Y
    ls_x = data[2:, 4]/X
    ls_y = data[2:, 5]/Y
    plt_title = model+' t1-t2 scores'
    if y_label ==  'id':
        plt_title = model+' t1 score Vs. pore depth'
        xlim = [-3.5, 3.5]
        ylim = [-300, 3200]
        y_label = 'id'
        

if model == 'SIMCA':
    data_bp = genfromtxt("data/"+model+" - Beadpack"+".csv",delimiter=',')
    data_ss = genfromtxt("data/"+model+" - Sandstone"+".csv",delimiter=',')
    data_ls = genfromtxt("data/"+model+" - Limestone"+".csv",delimiter=',')
    bp_x = data_bp[2:, 0]/X
    bp_y = data_bp[2:, 1]/Y
    ss_x = data_ss[2:, 0]/X
    ss_y = data_ss[2:, 1]/Y
    ls_x = data_ls[2:, 0]/X
    ls_y = data_ls[2:, 1]/Y
    plt_title = model+' t1-t2 scores'
    

print("Beadpack X range:("+str(min(bp_x))+", "+str(max(bp_x))+")")
print("Beadpack Y range:("+str(min(bp_y))+", "+str(max(bp_y))+")")
print("Sandstone X range:("+str(min(ss_x))+", "+str(max(ss_x))+")")
print("Sandstone Y range:("+str(min(ss_y))+", "+str(max(ss_y))+")")
print("Limestone X range:("+str(min(ls_x))+", "+str(max(ls_x))+")")
print("Limestone Y range:("+str(min(ls_y))+", "+str(max(ls_y))+")")


alpha = 0.9
# 'ggplot'
# 'seaborn-whitegrid'
# 'bmh'

# with plt.style.context(('seaborn-whitegrid')):
if x_label == 't1': 
    plt.figure(1)
    # plt.axhline(0, color='k', lw=1.5, alpha=0.3)
    # plt.axvline(0, color='k', lw=1.5, alpha=0.3)
    # leg = plt.legend()
    # leg.get_frame().set_edgecolor('k')
    # plt.axvline(x=0, color='k', linestyle='dashed',linewidth=0.7)
    # plt.axhline(y=0, color='k', linestyle='dashed',linewidth=0.7)
    plt.rc('axes', axisbelow=True)
    plt.rc('grid', linestyle="dotted", c='k',linewidth=0.5)
    plt.rc('axes', axisbelow=True)
    plt.grid(True)
    plt.scatter(bp_x, bp_y, c='g', edgecolors='k', linewidth=0.5, alpha=alpha, label="Synthetic Rock")
    plt.scatter(ss_x, ss_y, c='b', edgecolors='k', linewidth=0.5, alpha=alpha, label="Sandstone")
    plt.scatter(ls_x, ls_y, c='r', edgecolors='k', linewidth=0.5, alpha=alpha, label="Limestone")

    if model == 'PLS' or model =='PCA' or model == 'SIMCA' or y_label == 'pore depth':
        plt.xlim(xlim)
        plt.ylim(ylim)
    #plt.plot([70, 70], [100, 250], 'k-', lw=2)
    # tick.label.set_fontsize(10)
    # tick.label.set_rotation('vertical')
    # plt.xticks(fontsize=14, rotation=90)
    plt.title(plt_title, fontsize=14)
    plt.xlabel(x_label, color='k', fontsize=14)
    plt.ylabel(y_label, color='k', fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig('plots/'+plt_title+'.png', dpi=300, bbox_inches='tight')
    plt.show()