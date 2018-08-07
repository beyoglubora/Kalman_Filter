#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 18:49:44 2018

@author: Bora
"""
import numpy as np
from Functions.E_step import E_step
from Functions.M_step import M_step

def EM(y, p, num_iter):
    #intialize Kalman filter parameters
    A = np.identity(p)
    Q = np.identity(p)*(10**-3)
    e = 10**-2
    
    a_predicted = np.zeros((len(y), p, 1))
    a_refined = np.zeros((len(y), p, 1))
    
    P_predicted = np.zeros((len(y), p, p))
    P_refined = np.zeros((len(y), p, p))
    
    #initialize Kalman smoother parameters
    J = np.zeros((len(y), p, p))
    
    #initialize E-step parameters
    S_1 = np.zeros((len(y), p, p))
    S_2 = np.zeros((len(y), p, p))
    
    T = np.arange(p, len(y)) #start p ahead
    
    i = 0
    while i < num_iter:
        a_refined, P_refined, S_1, S_2 = E_step(a_predicted, a_refined, P_predicted, P_refined, A, Q, e, T, y, p, J, S_1, S_2)
        A, Q = M_step(S_1, S_2, A, a_refined, T, y, p)
        i = i + 1
        
    return a_refined, P_refined