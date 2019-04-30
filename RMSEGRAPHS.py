#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:34:30 2019

@author: VinayNair
"""
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

RegMod = ('Linear Regression', 'Polynomial Regression', 'Random Forrest Regression')
y_pos = np.arange(len(RegMod))
RSMEVals = [10555.3260546,11509.6925902,11210.4491777]

plt.bar(y_pos, RSMEVals, align='center', width=0.5, color=['cyan', 'red', 'green',], alpha = 0.5)
plt.xticks(y_pos, RegMod)
plt.ylabel('RSME Values')
plt.title('Performance of Regression Models')

plt.show()