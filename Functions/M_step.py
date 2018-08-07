#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 19:18:04 2018

@author: Bora
"""
import numpy as np
from numpy.linalg import inv
def M_step(S_1, S_2, A, a_refined, T, y, p):
    new_S_1 = np.sum(S_1[1:, :, :], axis = 0)
    new_S_2 = np.sum(S_2[1:, :, :], axis = 0)
    
    A_new = new_S_2 @ inv(new_S_1)
    Q_new = (1/(len(y)-1)) * (new_S_1 - (A_new @ new_S_2))
    
    summand = 0
    for t in T:
        h_y = y[t-p:t].reshape(5,1)
        summand = summand + (y[t]**2 - (2 * h_y.T @ a_refined[t, :] * y[t]) + (a_refined[t, :].T @ S_1[t, :, :] @ a_refined[t, :]))
    e_new = ((1/len(y)) * summand)[0][0]
    
    return A_new, Q_new