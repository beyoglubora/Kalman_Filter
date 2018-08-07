#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 18:09:11 2018

@author: Bora
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_model(initial_points, a_refined, y, T, p, savepath):
    points = np.zeros((len(T)+p,1))
    points[:p] = initial_points.reshape(p,1)
    for t in T:
        h_y = y[t-p:t].reshape(1,p)
        points[t, :] = h_y @ a_refined[t, :]
    
    plt.figure(figsize = (50, 25))
    plt.plot(y, 'ro')
    plt.plot(points, 'b-')
    plt.savefig(savepath)
    plt.legend('data', 'model')
    plt.close()
    