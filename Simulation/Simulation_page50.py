"""
MAIN CODE
Created on Sat Jun 12 19:09:06 2021

The graphics obtained in this simulation are presented in Chapter 6, page 50.

@author: katharinaenin
Isothermal Euler and Weymouth Equation with Lax Friedrich Scheme
and Simple Upwind.

"""
import numpy as np
import matplotlib.pyplot as plt

a_square = 340*340
T = 5
Lambda = 0.11
LengthOfPipe = 12000
D = 1

n = 6
m = 600

dt = T/(m)
dx = LengthOfPipe/(n-1)

P = np.zeros((n,m))
Q = np.zeros((n,m))

steps_Q = 10
init_Q = 550
for j in range(n):
    Q[j,0] = init_Q - steps_Q*j


steps_P = 3
init_P = 60
for j in range(n):
    P[j,0] = init_P - steps_P*j

Q[0,:] = Q[0,0]
P[0,:] = P[0,0]

X_Arr = np.arange(0,12,2)
T_Arr = np.arange(0,T,dt)

# Weymouth Gleichung mit LxF
def Weymouth_LxF():
    
    P1 = np.zeros((n,m))
    Q1 = np.zeros((n,m))
    P1[0,:] = P[0,0]
    Q1[0,:] = Q[0,0]
    P1[:,0] = P[:,0]
    Q1[:,0] = Q[:,0]
    Q1[n-1,:] = Q[n-1,:]
    P1[n-1,:] = P[n-1,:]
    for t in range(0,m-1):
        for j in range(1,n-1):
            P1[j,t+1] = 0.5*(P1[j+1,t]+P1[j-1,t]) - (dt/(2*dx))*(Q1[j+1,t]-Q1[j-1,t])
            Q1[j,t+1] = 0.5*(Q1[j+1,t]+Q1[j-1,t]) - a_square*(dt/(2*dx))*(P1[j+1,t]-P1[j-1,t]) - dt*Lambda/(4*D)*(Q1[j+1,t]*abs(Q1[j+1,t])/P1[j+1,t] + Q1[j-1,t]*abs(Q1[j-1,t])/P1[j-1,t])
            
    return P1, Q1

# Weymouth Gleichung mit Simple Upwind
def Weymouth_SimpleUp():

    P2 = np.zeros((n,m))
    Q2 = np.zeros((n,m))
    P2[0,:] = P[0,0]
    Q2[0,:] = Q[0,0]
    P2[:,0] = P[:,0]
    Q2[:,0] = Q[:,0]
    
    for t in range(0,m-1):
        for j in range(1,n):
            P2[j,t+1] = P2[j,t] - (dt/dx)*(Q2[j,t]-Q2[j-1,t])
            Q2[j,t+1] = Q2[j,t] - a_square*(dt/dx)*(P2[j,t]-P2[j-1,t]) - dt*Lambda/(2*D)*(Q2[j,t]*abs(Q2[j,t])/P2[j,t])
            
    P[n-1,:] = P2[n-1,:]       
    Q[n-1,:] = Q2[n-1,:] 

    return P2, Q2

# Euler Gleichung mit Simple Upwind
def EulerGleichung_SimpleUp():

    P3 = np.zeros((n,m))
    Q3 = np.zeros((n,m))
    P3[0,:] = P[0,0]
    Q3[0,:] = Q[0,0]
    P3[:,0] = P[:,0]
    Q3[:,0] = Q[:,0]
    
    for t in range(0,m-1):
        for j in range(1,n):
            P3[j,t+1] = P3[j,t] - (dt/dx)*(Q3[j,t]-Q3[j-1,t])
            Q3[j,t+1] = Q3[j,t] - (dt/dx)*(Q3[j,t]*Q3[j,t]/P3[j,t] + a_square*P3[j,t] - Q3[j-1,t]*Q3[j-1,t]/P3[j-1,t] - a_square*P3[j-1,t]) - dt*Lambda/(2*D)*(Q3[j,t]*abs(Q3[j,t])/P3[j,t])
    
    return P3, Q3

# Euler Gleichung mit LxF
def EulerGleichung_LxF():
    P4 = np.zeros((n,m))
    Q4 = np.zeros((n,m))
    P4[0,:] = P[0,0]
    Q4[0,:] = Q[0,0]
    P4[:,0] = P[:,0]
    Q4[:,0] = Q[:,0]
    Q4[n-1,:] = Q[n-1,:]
    P4[n-1,:] = P[n-1,:]
    
    for t in range(0,m-1):
        for j in range(1,n-1):
            P4[j,t+1] = 0.5*(P4[j-1,t] + P4[j+1,t]) - (dt/(2*dx))*(Q4[j+1,t]-Q4[j-1,t])
            Q4[j,t+1] = 0.5*(Q4[j-1,t]+Q4[j+1,t]) - (dt/(2*dx))*(Q4[j+1,t]*Q4[j+1,t]/P4[j+1,t] + a_square*P4[j+1,t] - Q4[j-1,t]*Q4[j-1,t]/P4[j-1,t] - a_square*P4[j-1,t]) - dt*Lambda/(4*D)*(Q4[j-1,t]*abs(Q4[j-1,t])/P4[j-1,t]+Q4[j+1,t]*abs(Q4[j+1,t])/P4[j+1,t])
    
    return P4, Q4
    
if __name__ == '__main__':
    EulerPlt = 1
    WeymouthBlt = 1
    SimpleUp_Euler_Blt = 1
    SimpleUp_Weymouth_Blt = 1
    
    if SimpleUp_Weymouth_Blt:
        P1,Q1 = Weymouth_SimpleUp()
        
        X_mesh, T_mesh = np.meshgrid(X_Arr, T_Arr)
        P_Matrix_T = P1.transpose()
        Q_Matrix_T = Q1.transpose()
        
        plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.dist = 11
        ax.set_xlabel('x in $10^3$ (m)')
        ax.set_ylabel('t')
        ax.set_zlabel(r'$\rho(t,x)$')
        ax.locator_params(axis='x', nbins = 6)
        ax.locator_params(axis='y', nbins = 5)
        ax.locator_params(axis='z', nbins = 5)
        ax.plot_surface(X_mesh, T_mesh, P_Matrix_T)
        title = 'Weymouth Equation with Simple Upwind: rho(t,x)'
        ax.set_title(title)
        # title_file1 = 'Wey_Simpl_P.png'
        # plt.savefig(title_file1, dpi=300)
        
        plt.figure(2)
        ax2 = plt.axes(projection='3d')
        ax2.dist = 11
        ax2.set_xlabel('x in $10^3$ (m)')
        ax2.set_ylabel('t')
        ax2.set_zlabel('q(t,x)')
        ax2.locator_params(axis='x', nbins = 6)
        ax2.locator_params(axis='y', nbins = 5)
        ax2.locator_params(axis='z', nbins = 5)
        ax2.plot_surface(X_mesh, T_mesh, Q_Matrix_T)
        title = 'Weymouth Equation with Simple Upwind: q(t,x)'
        ax2.set_title(title)
        # title_file2 = 'Wey_Simpl_Q.png'
        # plt.savefig(title_file2, dpi=300)
    
    if WeymouthBlt:
        P2, Q2 = Weymouth_LxF()
        
        X_mesh, T_mesh = np.meshgrid(X_Arr, T_Arr)
        P_Matrix_T = P2.transpose()
        Q_Matrix_T = Q2.transpose()
        
        plt.figure(3)
        ax = plt.axes(projection='3d')
        ax.dist = 11
        ax.set_xlabel('x in $10^3$ (m)')
        ax.set_ylabel('t')
        ax.set_zlabel(r'$\rho(t,x)$')
        ax.locator_params(axis='x', nbins = 6)
        ax.locator_params(axis='y', nbins = 5)
        ax.locator_params(axis='z', nbins = 5)
        ax.plot_surface(X_mesh, T_mesh, P_Matrix_T)
        title = 'Weymouth Equation with LxF: rho(t,x)'
        ax.set_title(title)
        # title_file1 = 'Wey_LxF_P.png'
        # plt.savefig(title_file1, dpi=300)
        
        plt.figure(4)
        ax2 = plt.axes(projection='3d')
        ax2.dist = 11
        ax2.set_xlabel('x in $10^3$ (m)')
        ax2.set_ylabel('t')
        ax2.set_zlabel('q(t,x)')
        ax2.locator_params(axis='x', nbins = 6)
        ax2.locator_params(axis='y', nbins = 5)
        ax2.locator_params(axis='z', nbins = 5)
        ax2.plot_surface(X_mesh, T_mesh, Q_Matrix_T)
        title = 'Weymouth Equation with LxF: q(t,x)'
        ax2.set_title(title)
        # title_file2 = 'Wey_LxF_Q.png'
        # plt.savefig(title_file2, dpi=300)
        
    if SimpleUp_Euler_Blt:
        P3, Q3 = EulerGleichung_SimpleUp()
        
        X_mesh, T_mesh = np.meshgrid(X_Arr, T_Arr)
        P_Matrix_T = P3.transpose()
        Q_Matrix_T = Q3.transpose()
        
        plt.figure(5)
        ax = plt.axes(projection='3d')
        ax.dist = 11
        ax.set_xlabel('x in $10^3$ (m)')
        ax.set_ylabel('t')
        ax.set_zlabel(r'$\rho(t,x)$')
        ax.locator_params(axis='x', nbins = 6)
        ax.locator_params(axis='y', nbins = 5)
        ax.locator_params(axis='z', nbins = 5)
        ax.plot_surface(X_mesh, T_mesh, P_Matrix_T)
        title = 'Euler Equation with Simple Upwind: rho(t,x)'
        ax.set_title(title)
        # title_file1 = 'Euler_Simpl_P.png'
        # plt.savefig(title_file1, dpi=300)
        
        plt.figure(6)
        ax2 = plt.axes(projection='3d')
        ax2.dist = 11
        ax2.set_xlabel('x in $10^3$ (m)')
        ax2.set_ylabel('t')
        ax2.set_zlabel('q(t,x)')
        ax2.locator_params(axis='x', nbins = 6)
        ax2.locator_params(axis='y', nbins = 5)
        ax2.locator_params(axis='z', nbins = 5)
        ax2.plot_surface(X_mesh, T_mesh, Q_Matrix_T)
        title = 'Euler Equation with Simple Upwind: q(t,x)'
        ax2.set_title(title)
        # title_file2 = 'Euler_Simpl_Q.png'
        # plt.savefig(title_file2, dpi=300)
    
    if EulerPlt:
        P4, Q4 = EulerGleichung_LxF()
        
        X_mesh, T_mesh = np.meshgrid(X_Arr, T_Arr)
        P_Matrix_T = P4.transpose()
        Q_Matrix_T = Q4.transpose()
        
        plt.figure(7)
        ax = plt.axes(projection='3d')
        ax.dist = 11
        ax.set_xlabel('x in $10^3$ (m)')
        ax.set_ylabel('t')
        ax.set_zlabel(r'$\rho(t,x)$')
        ax.locator_params(axis='x', nbins = 6)
        ax.locator_params(axis='y', nbins = 5)
        ax.locator_params(axis='z', nbins = 5)
        ax.plot_surface(X_mesh, T_mesh, P_Matrix_T)
        title = 'Euler Equation with LxF: rho(t,x)'
        ax.set_title(title)
        # title_file1 = 'Euler_LxF_P.png'
        # plt.savefig(title_file1, dpi=300)
        
        plt.figure(8)
        ax2 = plt.axes(projection='3d')
        ax2.dist = 11
        ax2.set_xlabel('x in $10^3$ (m)')
        ax2.set_ylabel('t')
        ax2.set_zlabel('q(t,x)')
        ax2.locator_params(axis='x', nbins = 6)
        ax2.locator_params(axis='y', nbins = 5)
        ax2.locator_params(axis='z', nbins = 5)
        ax2.plot_surface(X_mesh, T_mesh, Q_Matrix_T)
        title = 'Euler Equation with LxF: q(t,x)'
        ax2.set_title(title)
        # title_file2 = 'Euler_LxF_Q.png'
        # plt.savefig(title_file2, dpi=300)
        
        