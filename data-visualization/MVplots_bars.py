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

data_file = "OPLS VIP Pred"

data = genfromtxt("data/"+data_file+".csv",delimiter=',')
   
y = data[2:, 1]
x = ['Porosity',
     'Convexity',
     'nu11',
     'P/A',
     'nu02',
     'Area(m00)',
     'm02',
     'm01',
     'n03',
     'm11',
     'm03',
     'm10',
     'm12',
     'm30',
     'm21',
     'm20',
     'mu03',
     'mu21',
     'mu02',
     'mu11',
     'mu30',
     'nu12',
     'mu12',
     'nu30',
     'nu21',
     'mu20',
     'nu20']


with plt.style.context(('seaborn-whitegrid')):
    plt.figure(1, figsize = (30, 10))
    plt.rc('axes', axisbelow=True, ec='k', linewidth=1.5)
    plt.rc('grid', linestyle="dotted", c='k', linewidth=1)
    plt.grid(True)
    plt.bar(x, y, color ='blue',  width = 0.8, alpha=0.9)
    plt.yticks(fontsize=35) 
    plt.xticks(fontsize=35, rotation=90)
    # plt.xlabel("Variables", fontsize=30)
    # plt.ylabel("Variable Importance in Projection (VIP) scores", fontsize=12)
    plt.ylabel("VIP Scores\n", fontsize=35)
    plt_title = "VIP Predictions"
    # plt.title(plt_title , fontsize=14)
    plt.savefig('plots/'+plt_title+'1.png', show_shapes=True, dpi=200, bbox_inches='tight')
    plt.show()

