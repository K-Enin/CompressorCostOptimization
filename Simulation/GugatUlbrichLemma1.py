"""
MAIN CODE
Created on Sat Jun 12 19:09:06 2021

The graphics obtained in this simulation are presented in Chapter 6.

@author: katharinaenin
Isothermal Euler and Weymouth Equation with Lax Friedrich Scheme
Initial Conditions: Lemma 1 in Paper "The isothermal Euler equation for ideal gas
with source term".

The purpose of this code is to compare the LxF solution to the graphics in
the paper.
Simulation need quite some time!
"""
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from mpl_toolkits import mplot3d

a_square = 1
teta = 0.7
# CFL = 0.1
T = 2
X = 2.5
dx = 0.0625
n = 40 # n = 2.5/0.0625 (space steps)
beta = -1

# Euler Gleichung mit LxF
def EulerGleichung_LxF():
    # initial conditions
    P_Euler = np.zeros((n+1,1))
    P_Euler_new = np.zeros((n+1,1))
    
    x = 0
    X_array = [x]
    for i in range(0,n+1):
        P_Euler[i,0] = exp(beta*x) # t = 0
        x = x + dx
        X_array = np.append(X_array,x)
        
    Q_Euler = np.zeros((n+1,1))
    Q_Euler_new = np.zeros((n+1,1))      
            
    P_Matrix = P_Euler
    Q_Matrix = Q_Euler
    
    T = 0
    GeneralSteps = 0
    T_array = [T]
    
    while T < 2:
        # Another way of determining dt: Calculate dt with the CFL condition
        # dt = (CFL*dx)/(np.amax(abs(Q_Euler/P_Euler) + a_square*np.ones((1,n+1))))
        dt = 0.0001
        
        # Get the time value 
        T = T + dt
        T_array = np.append(T_array, T)
        
        for j in range(1,n):
            P_Euler_new[j,0] = 0.5*(P_Euler[j-1,0]+P_Euler[j+1,0]) - dt/(2*dx)*(Q_Euler[j+1,0]-Q_Euler[j-1,0])
            Q_Euler_new[j,0] = 0.5*(Q_Euler[j-1,0]+Q_Euler[j+1,0]) - dt/(2*dx)*((Q_Euler[j+1,0]*Q_Euler[j+1,0])/(P_Euler[j+1,0]) + a_square*P_Euler[j+1,0] - (Q_Euler[j-1,0]*Q_Euler[j-1,0])/(P_Euler[j-1,0])-a_square*P_Euler[j-1,0]) - dt*teta/4*(Q_Euler[j-1,0]*abs(Q_Euler[j-1,0])/P_Euler[j-1,0] + Q_Euler[j+1,0]*abs(Q_Euler[j+1,0])/P_Euler[j+1,0]) 
        
        # Compatibility conditions
        P_Euler_new[0,0] = 4.75*T*T+1 
        P_Euler_new[n,0] = P_Euler[n,0]

        Q_Euler_new[0,0] = 10*T*T
        Q_Euler_new[n,0] = Q_Euler[n,0]

        P_Euler = P_Euler_new
        Q_Euler = Q_Euler_new
        
        P_Matrix = np.append(P_Matrix, P_Euler, axis = 1)
        Q_Matrix = np.append(Q_Matrix, Q_Euler, axis = 1)
        
        # print every 1000 steps
        GeneralSteps = GeneralSteps + 1
        
        if GeneralSteps%1000 == 0:
            print("General Step:" + str(GeneralSteps) + " ,Time: " + str(T))
            
    return dt, GeneralSteps, X_array[:-1], T_array, P_Matrix, Q_Matrix


# Weymouth Gleichung mit LxF
def Weymouth_LxF():
    # initial conditions
    P_Euler = np.zeros((n+1,1))
    P_Euler_new = np.zeros((n+1,1))
    
    x = 0
    X_array = [x]
    for i in range(0,n+1):
        P_Euler[i,0] = exp(beta*x)
        x = x + dx
        X_array = np.append(X_array,x)
        
    Q_Euler = np.zeros((n+1,1))
    Q_Euler_new = np.zeros((n+1,1))      
            
    P_Matrix = P_Euler
    Q_Matrix = Q_Euler
    
    T = 0
    GeneralSteps = 0
    T_array = [T]
    
    # use the same step as in Euler LxF
    dt = 0.0001
    
    while T < 2:
        # Get the time value 
        T = T + dt
        T_array = np.append(T_array, T)
        
        for j in range(1,n):
            P_Euler_new[j,0] = 0.5*(P_Euler[j-1,0]+P_Euler[j+1,0]) - dt/(2*dx)*(Q_Euler[j+1,0]-Q_Euler[j-1,0])
            Q_Euler_new[j,0] = 0.5*(Q_Euler[j-1,0]+Q_Euler[j+1,0]) - dt/(2*dx)*a_square*(P_Euler[j+1,0]-P_Euler[j-1,0]) - dt*teta/4*(Q_Euler[j-1,0]*abs(Q_Euler[j-1,0])/P_Euler[j-1,0] + Q_Euler[j+1,0]*abs(Q_Euler[j+1,0])/P_Euler[j+1,0]) 

        # Compatibility conditions
        P_Euler_new[0,0] = 4.75*T*T+1
        P_Euler_new[n,0] = P_Euler[n,0]

        Q_Euler_new[0,0] = 10*T*T
        Q_Euler_new[n,0] = Q_Euler[n,0]

        P_Euler = P_Euler_new
        Q_Euler = Q_Euler_new
        
        P_Matrix = np.append(P_Matrix, P_Euler, axis = 1)
        Q_Matrix = np.append(Q_Matrix, Q_Euler, axis = 1)
        
        # print every 1000 steps
        GeneralSteps = GeneralSteps + 1
        if GeneralSteps%1000 == 0:
            print("General Step:" + str(GeneralSteps) + " ,Time: " + str(T))
            
    return dt, GeneralSteps, X_array[:-1], T_array, P_Matrix, Q_Matrix


# Euler Gleichung mit Simple Upwind
def EulerGleichung_SimpleUp():
    # initial conditions
    P_Euler = np.zeros((n+1,1))
    P_Euler_new = np.zeros((n+1,1))
    
    x = 0
    X_array = [x]
    for i in range(0,n+1):
        P_Euler[i,0] = exp(beta*x)
        x = x + dx
        X_array = np.append(X_array,x)
        
    Q_Euler = np.zeros((n+1,1))
    Q_Euler_new = np.zeros((n+1,1))      
            
    P_Matrix = P_Euler
    Q_Matrix = Q_Euler
    
    T = 0
    GeneralSteps = 0
    T_array = [T]
    
    while T < 2:
        dt = 0.0001
        
        # Get the time value 
        T = T + dt
        T_array = np.append(T_array, T)
        
        for j in range(1,n+1):
            P_Euler_new[j,0] = P_Euler[j,0] + dt/dx*(Q_Euler[j-1,0]-Q_Euler[j,0])
            Q_Euler_new[j,0] = Q_Euler[j,0] + dt/dx*((Q_Euler[j-1,0]*Q_Euler[j-1,0])/(P_Euler[j-1,0]) + a_square*P_Euler[j-1,0] - (Q_Euler[j,0]*Q_Euler[j,0])/(P_Euler[j,0])-a_square*P_Euler[j,0]) - dt*teta/2*(Q_Euler[j,0]*abs(Q_Euler[j,0])/P_Euler[j,0]) 
        
        # Compatibility conditions
        P_Euler_new[0,0] = 4.75*T*T+1
        #P_Euler_new[n,0] = P_Euler[n,0]         #not needed here because Simple Up calculates this 

        Q_Euler_new[0,0] = 10*T*T
        #Q_Euler_new[n,0] = Q_Euler[n,0]         #not needed here because Simple Up calculates this 

        P_Euler = P_Euler_new
        Q_Euler = Q_Euler_new
        
        P_Matrix = np.append(P_Matrix, P_Euler, axis = 1)
        Q_Matrix = np.append(Q_Matrix, Q_Euler, axis = 1)
        
        # print every 1000 general steps
        GeneralSteps = GeneralSteps + 1
        
        if GeneralSteps%1000 == 0:
            print("General Step:" + str(GeneralSteps) + " ,Time: " + str(T))

    return dt, GeneralSteps, X_array[:-1], T_array, P_Matrix, Q_Matrix


# Euler Gleichung mit Simple Upwind
def Weymouth_SimpleUp():
    # initial conditions
    P_Euler = np.zeros((n+1,1)) 
    P_Euler_new = np.zeros((n+1,1)) 
    
    x = 0
    X_array = [x]
    for i in range(0,n+1):
        P_Euler[i,0] = exp(beta*x)
        x = x + dx
        X_array = np.append(X_array,x)
        
    Q_Euler = np.zeros((n+1,1))
    Q_Euler_new = np.zeros((n+1,1))      
            
    P_Matrix = P_Euler
    Q_Matrix = Q_Euler
    
    T = 0
    GeneralSteps = 0
    T_array = [T]
    
    dt = 0.0001
    
    while T < 2:
        
        # Get the time value 
        T = T + dt
        T_array = np.append(T_array, T)
        
        for j in range(1,n+1):
            P_Euler_new[j,0] = P_Euler[j,0] - dt/dx*(Q_Euler[j,0]-Q_Euler[j-1,0])
            Q_Euler_new[j,0] = Q_Euler[j,0] - dt/dx*a_square*(P_Euler[j,0]-P_Euler[j-1,0]) - dt*teta/2*(Q_Euler[j,0]*abs(Q_Euler[j,0])/P_Euler[j,0]) 
        
        # Compatibility conditions
        P_Euler_new[0,0] = 4.75*T*T+1
        #P_Euler_new[n,0] = P_Euler[n,0]         #not needed here because Simple Up calculates this 

        Q_Euler_new[0,0] = 10*T*T
        #Q_Euler_new[n,0] = Q_Euler[n,0]         #not needed here because Simple Up calculates this 

        P_Euler = P_Euler_new
        Q_Euler = Q_Euler_new
        
        P_Matrix = np.append(P_Matrix, P_Euler, axis = 1)
        Q_Matrix = np.append(Q_Matrix, Q_Euler, axis = 1)
        
        # print every 1000 general steps
        GeneralSteps = GeneralSteps + 1
        
        if GeneralSteps%1000 == 0:
            print("General Step:" + str(GeneralSteps) + " ,Time: " + str(T))
            
    return dt, GeneralSteps, X_array[:-1], T_array, P_Matrix, Q_Matrix

if __name__ == '__main__':
    EulerPlt = 1
    WeymouthBlt = 1
    SimpleUp_Euler_Blt = 1
    SimpleUp_Weymouth_Blt = 1
    
    if (EulerPlt == 1):
        # Plot Euler with LxF
        dt1, steps1, X_Arr, T_Arr, P_Matrix, Q_Matrix = EulerGleichung_LxF()
        X_mesh, T_mesh = np.meshgrid(X_Arr, T_Arr)
        P_Matrix_T = P_Matrix.transpose()
        Q_Matrix_T = Q_Matrix.transpose()
        
        plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.dist = 11
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('p(t,x)')
        ax.locator_params(axis='x', nbins = 6)
        ax.locator_params(axis='y', nbins = 5)
        ax.locator_params(axis='z', nbins = 5)
        ax.plot_surface(X_mesh, T_mesh, P_Matrix_T)
        title = 'Euler Equation with LxF: p(t,x)'
        ax.set_title(title)
        title_file1 = 'LxF_Euler_P_' + str(dt1) + '.png'
        plt.savefig(title_file1, dpi=300)
        
        plt.figure(2)
        ax2 = plt.axes(projection='3d')
        ax2.dist = 11
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('q(t,x)')
        ax2.locator_params(axis='x', nbins = 6)
        ax2.locator_params(axis='y', nbins = 5)
        ax2.locator_params(axis='z', nbins = 5)
        ax2.plot_surface(X_mesh, T_mesh, Q_Matrix_T)
        title = 'Euler Equation with LxF: q(t,x)'
        ax2.set_title(title)
        title_file2 = 'LxF_Euler_Q_' + str(dt1) + '.png'
        plt.savefig(title_file2, dpi=300)

    if (WeymouthBlt == 1):
        # Plot Weymouth with LxF
        dt2, steps2, X_Arr2, T_Arr2, P_Matrix2, Q_Matrix2 = Weymouth_LxF()
        X_mesh2, T_mesh2 = np.meshgrid(X_Arr2, T_Arr2)
        P_Matrix_T2 = P_Matrix2.transpose()
        Q_Matrix_T2 = Q_Matrix2.transpose()
        
        plt.figure(3)
        ax3 = plt.axes(projection='3d')
        ax3.dist = 11
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('p(t,x)')
        ax3.locator_params(axis='x', nbins = 6)
        ax3.locator_params(axis='y', nbins = 5)
        ax3.locator_params(axis='z', nbins = 5)
        ax3.plot_surface(X_mesh2, T_mesh2, P_Matrix_T2)
        title = 'Weymouth Equation with LxF: p(t,x)'
        ax3.set_title(title)
        title_file3 = 'LxF_Weymouth_P_' + str(dt2) + '.png'
        plt.savefig(title_file3, dpi=300)
        
        plt.figure(4)
        ax4 = plt.axes(projection='3d')
        ax4.dist = 11
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        ax4.locator_params(axis='x', nbins = 6)
        ax4.locator_params(axis='y', nbins = 5)
        ax4.locator_params(axis='z', nbins = 5)
        ax4.set_zlabel('q(t,x)')
        ax4.plot_surface(X_mesh2, T_mesh2, Q_Matrix_T2)
        title = 'Weymouth Equation with LxF: q(t,x)'
        ax4.set_title(title)
        title_file4 = 'LxF_Weymouth_Q_' + str(dt2) + '.png'
        plt.savefig(title_file4, dpi=300)
        
    if (SimpleUp_Euler_Blt == 1):
        # Plot Euler with Simple Upwind
        dt3, steps3, X_Arr3, T_Arr3, P_Matrix3, Q_Matrix3 = EulerGleichung_SimpleUp()
        X_mesh3, T_mesh3 = np.meshgrid(X_Arr3, T_Arr3)
        P_Matrix_T3 = P_Matrix3.transpose()
        Q_Matrix_T3 = Q_Matrix3.transpose()
        
        plt.figure(5)
        ax5 = plt.axes(projection='3d')
        ax5.dist = 11
        ax5.set_xlabel('x')
        ax5.set_ylabel('t')
        ax5.set_zlabel('p(t,x)')
        ax5.locator_params(axis='x', nbins = 6)
        ax5.locator_params(axis='y', nbins = 5)
        ax5.locator_params(axis='z', nbins = 5)
        ax5.plot_surface(X_mesh3, T_mesh3, P_Matrix_T3)
        title = 'Euler Equation with Simple Upwind: p(t,x)'
        ax5.set_title(title)
        title_file5 = 'SimpleUp_Euler_P_' + str(dt3) + '.png'
        plt.savefig(title_file5, dpi=300)
        
        plt.figure(6)
        ax6 = plt.axes(projection='3d')
        ax6.dist = 11
        ax6.set_xlabel('x')
        ax6.set_ylabel('t')
        ax6.locator_params(axis='x', nbins = 6)
        ax6.locator_params(axis='y', nbins = 5)
        ax6.locator_params(axis='z', nbins = 5)
        ax6.set_zlabel('q(t,x)')
        ax6.plot_surface(X_mesh3, T_mesh3, Q_Matrix_T3)
        title = 'Euler Equation with Simple Upwind: q(t,x)'
        ax6.set_title(title)
        title_file6 = 'SimpleUp_Euler_Q_' + str(dt3) + '.png'
        plt.savefig(title_file6, dpi=300)
        
        
    if (SimpleUp_Weymouth_Blt == 1):
        # Plot WEymouth with Simple Upwind
        dt4, steps4, X_Arr4, T_Arr4, P_Matrix4, Q_Matrix4 = Weymouth_SimpleUp()
        X_mesh4, T_mesh4 = np.meshgrid(X_Arr4, T_Arr4)
        P_Matrix_T4 = P_Matrix4.transpose()
        Q_Matrix_T4 = Q_Matrix4.transpose()
        
        plt.figure(7)
        ax7 = plt.axes(projection='3d')
        ax7.dist = 11
        ax7.set_xlabel('x')
        ax7.set_ylabel('t')
        ax7.set_zlabel('p(t,x)')
        ax7.locator_params(axis='x', nbins = 6)
        ax7.locator_params(axis='y', nbins = 5)
        ax7.locator_params(axis='z', nbins = 5)
        ax7.plot_surface(X_mesh4, T_mesh4, P_Matrix_T4)
        title = 'Weymouth Equation with Simple Upwind: p(t,x)'
        ax7.set_title(title)
        title_file7 = 'SimpleUp_Weymouth_P_' + str(dt4) + '.png'
        plt.savefig(title_file7, dpi=300)
        
        plt.figure(8)
        ax8 = plt.axes(projection='3d')
        ax8.dist = 11
        ax8.set_xlabel('x')
        ax8.set_ylabel('t')
        ax8.locator_params(axis='x', nbins = 6)
        ax8.locator_params(axis='y', nbins = 5)
        ax8.locator_params(axis='z', nbins = 5)
        ax8.set_zlabel('q(t,x)')
        ax8.plot_surface(X_mesh4, T_mesh4, Q_Matrix_T4)
        title = 'Weymouth Equation with Simple Upwind: q(t,x)'
        ax8.set_title(title)
        title_file8 = 'SimpleUp_Weymouth_Q_' + str(dt4) + '.png'
        plt.savefig(title_file8, dpi=300)