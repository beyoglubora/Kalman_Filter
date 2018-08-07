#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 17:39:28 2018

@author: Bora
"""
import numpy as np
from numpy.linalg import inv
def kalman_filter(a_predicted, a_refined, P_predicted, P_refined, A, Q, e, T, y, p):
    I = np.identity(p)
    for t in T:
        #make initial predictions
        a_predicted[t, :] = A @ a_refined[t-1,:]
        P_predicted[t, :, :] = A @ P_refined[t-1, :, :] @ A.T + Q
        
        #get previous points to compute kalman gain
        h_y = y[t-p:t].reshape(1,p) 
        #compute kalman gain
        K = P_predicted[t, :, :] @ h_y.T * inv(e + h_y @ P_predicted[t, :, :] @ h_y.T)
        
        #refine the predictions
        a_refined[t, :] = a_refined[t-1,:] + K * (y[t] - h_y @ a_predicted[t, :])
        P_refined[t, :, :] = (I - K @ h_y) @ P_predicted[t, :, :] @ (I - K @ h_y).T + e * K @ K.T
    
    return a_predicted, a_refined, P_predicted, P_refined