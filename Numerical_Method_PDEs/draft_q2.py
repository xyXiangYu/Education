#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:28:09 2018

@author: xiang
"""

tvalue = np.arange(0, 1, 0.00001)

yvalue = 1./3 * np.exp(tvalue)

plt.plot(yvalue, tvalue)
