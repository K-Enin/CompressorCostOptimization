#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 16:25:35 2021

@author: katharinaenin
Example included in the Sum Up Rounding Chapter
Here Standrad Rounding is also considered
""" 
import numpy as np
import matplotlib.pyplot as plt


def standard_rounding(alpha):
    N, modes = alpha.shape 
    p = np.zeros((N, modes))
    for i in range(N):
        phat = alpha[i,:]
        j = np.argmax(phat)
        p[i,j] = 1.
    return p
    

def sum_up_rounding(alpha):
    'Sum-Up-Rounding with SOS1 constraint on equidistant grid'
    N, conf = alpha.shape
    step = 1
    p = np.zeros((N, conf))
    array = np.zeros(conf)
    unique = True
    for i in range(N):
        array += step * alpha[i,:]
        j = np.argmax(array) 
        if np.sum(array == array[j]) > 1: 
            unique = False
        p[i,j] = 1
        array[j] -= step
    if not unique:
        print('\nWarning: Sum-Up Rounding result not unique\n')
    return p

if __name__ == '__main__':
    numbers = np.arange(0,8)
    alpha = np.array([[0, 0.5, 0.25, 0.1,0.9,0.6,0.33,0.8], [1, 0.5, 0.75, 0.9, 0.1, 0.4,0.67,0.2]])
    alpha = alpha.transpose()
    alpha1 = standard_rounding(alpha)
    alpha2 = sum_up_rounding(alpha)
    
    plt.figure(1).clear()
    plt.bar(numbers, alpha[:,0], width = 1, edgecolor='k', color = 'lightcoral', alpha=0.7)
    plt.yticks(np.arange(0, 1.1, 0.5))
    plt.xlabel('time i')
    plt.title('Original values')
    
    plt.figure(2).clear()
    plt.bar(numbers, alpha1[:,0], width = 1, edgecolor='k', color = 'lightcoral', alpha=0.7)
    plt.yticks(np.arange(0, 1.1, 0.5))
    plt.xlabel('time i')
    plt.title('Values calculated by Standard Rounding')
    
    plt.figure(3).clear()
    plt.bar(numbers, alpha2[:,0], width = 1, edgecolor='k', color = 'lightcoral', alpha=0.7)
    plt.yticks(np.arange(0, 1.1, 0.5))
    plt.xlabel('time i')
    plt.title('Values calculated by Sum Up Rounding')
    