#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 18:29:21 2018

@author: Bora
"""
import numpy as np
from numpy.linalg import inv

def kalman_smoother(a_predicted, a_refined, P_predicted, P_refined, A, J, T):
    for t in T:
        J[t-1, :, :] = P_refined[t-1, :, :] @ A.T @ inv(P_predicted[t, :, :])
        a_refined[t-1, :] = a_refined[t-1, :] + (J[t-1, :, :] @ (a_refined[t, :] - (A @ a_refined[t-1, :])))
        P_refined[t-1, :] = P_refined[t-1, :, :] + (J[t-1, :, :] @ (P_refined[t, :, :] - P_predicted[t, :, :]) @ J[t-1, :, :].T)
    return a_refined, P_refined, J