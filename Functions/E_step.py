#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 18:55:40 2018

@author: Bora
"""
import numpy as np
from Functions.kalman_filter import kalman_filter
from Functions.kalman_smoother import kalman_smoother

def E_step(a_predicted, a_refined, P_predicted, P_refined, A, Q, e, T, y, p, J, S_1, S_2):
    #call kalman_filter
    a_predicted, a_refined, P_predicted, P_refined = kalman_filter(a_predicted, a_refined, P_predicted, P_refined, A, Q, e, T, y, p)
    
    #call kalman smoother
    T_reversed = np.flip(T, axis = 0)
    a_refined, P_refined, J = kalman_smoother(a_predicted, a_refined, P_predicted, P_refined, A, J, T_reversed)
    
    #compute expectation values
    for t in T:
        S_1[t, :, :] = P_refined[t, :, :] + (a_refined[t, :] @ a_refined[t, :].T)
        S_2[t, :, :] = (J[t-1, :, :] @ P_refined[t, :, :]) + (a_refined[t, :] @ a_refined[t-1, :].T)
    
    return a_refined, P_refined, S_1, S_2