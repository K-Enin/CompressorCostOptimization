#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:22:14 2021
@author: katharinaenin
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io

# Set condition case
case = "AdvancedModel"

if case == "SimpleModel":
    folder = 'Example20-Simple/'
    print("Data generation for a Simple Model.")
    n = 6
    m = 300
    number_of_edges = 3
    P_time0 = np.zeros((number_of_edges,n))
    Q_time0 = np.zeros((number_of_edges,n))
    P_time0[0,:] = 60
    P_time0[1,:] = 60
    P_time0[2,:] = 60
    Q_time0[:,:] = 500
    
    # no flow taken out of slack bus
    eps = np.zeros((1,m))
    np.savetxt(folder + 'eps_file.dat', eps)
    np.savetxt(folder + 'P_time0.dat', P_time0)
    np.savetxt(folder + 'Q_time0.dat', Q_time0)
elif case == "AdvancedModel":
    folder = 'Example_Advanced/'
    print("Data generation for an Advanced Model.")
    n = 6
    m = 300
    number_of_edges = 8
    
    P_time0 = np.zeros((number_of_edges,n))
    Q_time0 = np.zeros((number_of_edges,n))
    P_time0[0,:] = 60
    P_time0[1,:] = 62
    P_time0[2,:] = 62
    P_time0[3,:] = P_time0[2,:]
    P_time0[4,:] = 62
    P_time0[5,:] = P_time0[4,:]
    P_time0[6,:] = 62
    P_time0[7,:] = 62
    
    mat_file = scipy.io.loadmat(folder + 'eps_file.mat')
    eps = mat_file["eps"]
    eps[0,0] = round(eps[0,0],4) # to simplify summation
    
    Q_time0[0,:] = 500
    Q_time0[1,:] = 500
    Q_time0[2,:] = 250
    Q_time0[3,:] = 250
    Q_time0[4,:] = 250 - eps[0,0]
    Q_time0[5,:] = Q_time0[3,:]
    Q_time0[6,:] = Q_time0[4,:] + Q_time0[5,:]
    Q_time0[7,:] = Q_time0[6,:]
    
    np.savetxt(folder + 'P_time0.dat', P_time0)
    np.savetxt(folder + 'Q_time0.dat', Q_time0)

elif case == "AdvancedModel2":
    folder = 'Example_Advanced2/'
    print("Data generation for an Advanced Model.")
    n = 6
    m = 300
    number_of_edges = 8
    
        
    P_time0 = np.zeros((number_of_edges,n))
    Q_time0 = np.zeros((number_of_edges,n))
    
    # P for time 0 
    steps = (60 - 50)/(5 * n - 5)
    init = 60
    for i in [0,1,2,4,6,7]:
        for j in range(n):
            if j != 0:
                init = init - steps
            P_time0[i][j] = init
    P_time0[3,:] = P_time0[2,:]
    P_time0[5,:] = P_time0[4,:]
            
    # Q for time 0               
    steps_Q = (550 - 450)/(5 * n - 5) #550 - 400 war bisher das beste
    init_Q = 550
    for i in [0,1]:
        for j in range(n):
            if j != 0:
                init_Q = init_Q - steps_Q
            Q_time0[i][j] = init_Q
            
    mat_file = scipy.io.loadmat(folder + 'eps_file.mat')
    eps = mat_file["eps"]
    eps[0,0] = round(eps[0,0],4) # to simplify summation
    
    Q_time0[2,:] = Q_time0[1,n-1]/2 + eps[0,0]
    Q_time0[3,:] = Q_time0[1,n-1]/2 - eps[0,0]
    Q_time0[4,:] = Q_time0[2,n-1] - eps[0,0]
    Q_time0[5,:] = Q_time0[3,n-1]
    Q_time0[6,:] = (Q_time0[4,:] + Q_time0[5,:])
    Q_time0[7,:] = Q_time0[6,:]
    
    np.savetxt(folder + 'P_time0.dat', P_time0)
    np.savetxt(folder + 'Q_time0.dat', Q_time0)
else:
    print("No case provided.")