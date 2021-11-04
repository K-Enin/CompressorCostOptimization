#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25
Discretized NLP for compressor optimization with Bonmin (no POC reformulation).
@author: katharinaenin

"""
import casadi as cas
import configparser
import itertools
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from time import process_time
import math
import scipy.io 
import os

# Set folder name, provide like this: 'Example1/'
folder = 'Example_Advanced/'

# Additional constrains couple over time: True - on, False - offs
additional_constraints = False
if additional_constraints:
    M1, M2 = 1/12, 1/12
    
# Constraint on P and on Q
lbP = 0
# normale Pipes machen bis 150 bar mit
ubP = 100 
lbQ = -1000
ubQ = +1000

# Read Config Parameters in Configs.txt
config = configparser.ConfigParser()
config.read(folder + 'Configs.txt')

length_of_pipe = int(config['configs']['LengthPipe']) # here: universal length for all pipes (in m)
time_in_total = int(config['configs']['TimeInTotal']) # time (in sec)
Lambda = float(config['configs']['Lambda']) # here: Lambda is fix
D = int(config['configs']['Diameter']) # in m
min_pressure_last_node = int(config['configs']['MinPressureAtLastNode']) # in bar
a = int(config['configs']['FluxSpeed']) # in m/s
a_square = a*a
InitialGuess_P_beginEdge = int(config['InitialGuesses']['InitialGuess_P_beginEdge'])
InitialGuess_Q_beginEdge = int(config['InitialGuesses']['InitialGuess_Q_beginEdge'])
InitialGuess_P_middleEdge = int(config['InitialGuesses']['InitialGuess_P_middleEdge'])
InitialGuess_Q_middleEdge = int(config['InitialGuesses']['InitialGuess_Q_middleEdge'])
InitialGuess_P_endEdge = int(config['InitialGuesses']['InitialGuess_P_endEdge'])
InitialGuess_Q_endEdge = int(config['InitialGuesses']['InitialGuess_Q_endEdge'])


def df_to_int(df):
    """
    Function for extracting integers encapsulated in strings 
    into real integers, e.g. '3' becomes 3
    input: dataframe with all strings
    output: adjusted dataframe with int and strings
    """
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            try:
                df.iloc[i][j] = int(df.iloc[i][j])
            except:
                pass


def get_ingoing_edges(df, node):
    """
    Function for extracting ingoing edges for specific node
    input: dataframe, node from which we want to get the ingoing edges
    output: list of edges (int)
    """
    list_of_edges = []
    for i in range(df.shape[0]):
        if df.iloc[i][2] == node:
            list_of_edges.append(df.iloc[i][0])
                
    return list_of_edges


def get_outgoing_edges(df, node):
    """
    Function for extracting outgoing edges for specific node
    input: dataframe, node from which we want to get the outgoing edges
    output: list of edges (int)
    """
    list_of_edges = []
    for i in range(df.shape[0]):
        if df.iloc[i][1] == node:
            list_of_edges.append(df.iloc[i][0])

    return list_of_edges


def get_all_nodes(df):
    """
    Function for extracting all nodes from Edges.txt (inclusive slack & compressor)
    input: dataframe
    output: list of nodes (int & str)
    """
    list_of_all_nodes = []
    for i in range(df.shape[0]):
        for j in range(1,3):
            if df.iloc[i][j] not in list_of_all_nodes:
                list_of_all_nodes.append(df.iloc[i][j])

    return list_of_all_nodes


def get_end_node_in_network(df):
    """
    Function for extracting end node in network from Edges.txt
    assumption: there is only one end node, slack not included
    input: dataframe 
    output: last node in network (int)
    """
    all_nodes = get_all_nodes(df)
    end_nodes = []
    list_df2 = df.iloc[:,1].tolist()

    for node in all_nodes:
       if node not in list_df2:
           end_nodes.append(node)
    
    end_nodes.remove('s') # remove slack node
    
    return end_nodes


def get_end_edge_in_network(df):
    """
    Function for extracting last edge in network from Edges.txt
    assumption: there is only one end edge attached to single end node, 
    thus break for loop if it is found
    input: dataframe 
    output: last edge (int)
    """
    end_node = get_end_node_in_network(df)
    list_df2 = df.iloc[:,2].tolist()
    
    for i, node in enumerate(list_df2): 
        if node == end_node[0]:
            return df.iloc[i][0]


def get_starting_nodes_in_network(df):
    """
    Function for extracting starting nodes from Edges.txt
    input: dataframe 
    output: list of starting nodes (int)
    """
    starting_nodes = []
    all_nodes = get_all_nodes(df)
    list_df2 = df.iloc[:,2].tolist()

    for node in all_nodes:
        if node not in list_df2:
            starting_nodes.append(node)
    
    return starting_nodes


def get_starting_edges_in_network(df):
    """
    Function for extracting starting edges from Edges.txt
    input: dataframe 
    output: list of starting edges (int)
    """
    starting_edges = []
    list_df1 = df.iloc[:,1].tolist()
    starting_nodes = get_starting_nodes_in_network(df)

    for node in starting_nodes:
        for i in range(df.shape[0]):
            if list_df1[i] == node:
                starting_edges.append(df.iloc[i][0])
                break 
                # break, because starting node is only connected to one edge! 
    
    return starting_edges


def get_list_of_compressors(df):
    """
    Function for extracting all existing compressors from Edges.txt
    input: dataframe
    output: list of compressors (str)
    """
    list_of_compressors = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if(bool(re.match("c[0-9]",str(df.iloc[i][j])))):
                list_of_compressors.append(df.iloc[i][j])
    unique_list_of_compressors = list(set(list_of_compressors)) # make list unique
    
    return unique_list_of_compressors


def get_slack_connection_node(df):
    """
    Function for extracting the node which is connected to slack bus in Edges.txt
    assumption: only one node is connected to slack bus
    input: dataframe
    output: int
    """
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if df.iloc[i][j] == "s": # works only if 's' is written at out_node
                return df.iloc[i][j-1] # return in_node
            

def get_slack_connection_edge(df):
    """
    Function for extracting the edge which is connected to slack bus in Edges.txt
    assumption: only one edge is connected to slack bus
    input: dataframe
    output: int
    """
    slack_connection_node = get_slack_connection_node(df)
    list_of_outgoing_edges = get_outgoing_edges(df, slack_connection_node)
    for edge in list_of_outgoing_edges:
        if df.iloc[edge,2] == 's':
            return df.iloc[edge,0] # there is only one
        

def get_all_edges_without_slack_edge(df):
    """
    Function for returning list of edges without the edge connected to the slack bus
    (Useful for condition 2)
    input: dataframe
    output: list
    """
    list_df1 = df.iloc[:,2].tolist()
    list_of_edges = []
    for i, node in enumerate(list_df1):
        if node != 's' and i != 0:
            list_of_edges.append(df.iloc[i][0])
    
    return list_of_edges


def gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_lastnode, eps, Edgesfile):
    """
    Function for setting up the NLP
    input: P_time0, Q_time0, P_initialnode, Q_lastnode, eps
           Edgesfile as string file -> ('Edges.txt')
    output: nlp, lbw, ubw, lbg, ubg, w0
    """
    n = np.shape(P_time0)[1]       # Number of space steps
    print("This is n: " + str(n))
    m = np.shape(P_initialnode)[0] # Number of time steps
    print("This is m: " + str(m))
    dx = length_of_pipe/n # in (m)
    dt = time_in_total/m  # in (s) 
    
    # Sanity Checks
    # Is CFL fulfilled?
    if (dt*a_square > dx):
        raise Exception("CFL is not fulfilled.")
    # Dimensions are uniformy?
    if (np.shape(P_time0) != np.shape(Q_time0)):
        raise Exception("Mismatch of dimension of Q_time0, P_time0.")
    
    df = pd.read_csv(Edgesfile, header = None)
    df = df.iloc[1:, :] # drop first row, which is the header
    df = df.reset_index(drop=True)
    df_to_int(df)
    number_of_edges = df.shape[0]
    number_of_edges_without_slack_edge = number_of_edges - 1
    list_of_compressors = get_list_of_compressors(df)
    number_of_compressors = len(list_of_compressors)
    number_of_configs = 2**number_of_compressors
    slack_edge = get_slack_connection_edge(df)
    starting_edges = get_starting_edges_in_network(df)
    end_edge = get_end_edge_in_network(df)
    
    # variables
    w, w0, lbw, ubw  = [], [], [], []

    # constraints
    g, lbg, ubg = [], [], []  
    
    # Parameters we are looking for
    P, Q, u, beta = [], [], [], []
    # alpha does not need vector initialization
    
    # Vector provided information, whether value should be discrete or not
    discrete = []
    
    ################################
    ### Set initial conditions #####
    # u
    u = cas.MX.sym('u', m, number_of_compressors)
    w += [cas.reshape(u, -1, 1)]
    w0 += [0.] * m * number_of_compressors  
    lbw += [0.] * m * number_of_compressors
    ubw += [+cas.inf] * m * number_of_compressors
    discrete += [False] * m * number_of_compressors 
    
    # is u on or off
    beta = cas.MX.sym('u', m, number_of_compressors)
    w += [cas.reshape(beta, -1, 1)]
    w0 += [.5] * m * number_of_compressors
    lbw += [0.] * m * number_of_compressors
    ubw += [1.] * m * number_of_compressors 
    discrete += [True] * m * number_of_compressors 
    
    # P, Q 
    for edge in range(number_of_edges_without_slack_edge):
        P += [cas.MX.sym('P_{}'.format(edge), n, m)]
        Q += [cas.MX.sym('Q_{}'.format(edge), n, m)]
        w += [cas.reshape(P[edge], -1, 1), cas.reshape(Q[edge], -1, 1)]
        
        if edge in starting_edges: 
            # Go through P
            w0 += [*P_time0[edge]] + [InitialGuess_P_beginEdge]*(n)*(m-1)
            lbw += [*P_time0[edge]] + [lbP]*(n)*(m-1)
            ubw += [*P_time0[edge]] + [ubP]*(n)*(m-1)
            # Go through Q
            w0 += [*Q_time0[edge]] + [InitialGuess_Q_beginEdge]*(n)*(m-1)
            lbw += [*Q_time0[edge]] + [lbQ]*(n)*(m-1)
            ubw += [*Q_time0[edge]] + [ubQ]*(n)*(m-1)
            
        elif edge == end_edge: 
            # Go through P
            w0 += [*P_time0[edge]] + [InitialGuess_P_middleEdge]*(n)*(m-1)
            lbw += [*P_time0[edge]] + [lbP]*(n)*(m-1)
            ubw += [*P_time0[edge]] + [ubP]*(n)*(m-1)
            # Go through Q
            w0 += [*Q_time0[edge]] + [InitialGuess_Q_middleEdge]*(n)*(m-1)
            lbw += [*Q_time0[edge]] + [lbQ]*(n)*(m-1)
            ubw += [*Q_time0[edge]] + [ubQ]*(n)*(m-1)
        else: 
            # Go through P
            w0 += [*P_time0[edge]] + [InitialGuess_P_endEdge]*(n)*(m-1)
            lbw += [*P_time0[edge]] + [lbP]*(n)*(m-1)
            ubw += [*P_time0[edge]] + [ubP]*(n)*(m-1)
            # Go through Q
            w0 += [*Q_time0[edge]] + [InitialGuess_Q_endEdge]*(n)*(m-1)
            lbw += [*Q_time0[edge]] + [lbQ]*(n)*(m-1)
            ubw += [*Q_time0[edge]] + [ubQ]*(n)*(m-1)
    
    discrete += [False] * 2 * n * m * number_of_edges_without_slack_edge
    print("Initial conditions are set.")
    
    ####################
    #### Condition 1 ###
    # PDE constraint with Weymouth Equation
    bar_conv = 100000 # 1 bar = 100.000 (kg/(m*s^2)) (= 100.000 Pa)
    for edge in range(number_of_edges_without_slack_edge):
        for t in range(m-1):
            g += [P[edge][1:,t+1]*bar_conv/a_square - P[edge][1:,t]*bar_conv/a_square + (dt/dx)*(Q[edge][1:,t]-Q[edge][:-1,t])]
            lbg += [0.] * (n-1)
            ubg += [0.] * (n-1)
            
            g += [Q[edge][1:,t+1] - Q[edge][1:,t] + (dt/dx)*(P[edge][1:,t]*bar_conv - P[edge][:-1,t]*bar_conv) 
                  + a_square*dt*Lambda/(2*D)*(Q[edge][1:,t]*cas.fabs(Q[edge][1:,t]))/(P[edge][1:,t]*bar_conv)]
            lbg += [0.] * (n-1)
            ubg += [0.] * (n-1)
            
    print("Condition 1 is set.")
    
    ####################
    #### Condition 2 ###
    # Node property
    
    ### sum q_in = sum q_out
    nodes_list = get_all_nodes(df)
    slack_connection_node = get_slack_connection_node(df)
    starting_nodes = get_starting_nodes_in_network(df)
    end_node = get_end_node_in_network(df)
    
    # Filter out all unnecessary nodes from nodes_list 
    # which are starting nodes, ending nodes, slack attached nodes
    for node in starting_nodes:
        nodes_list.remove(node)
    nodes_list.remove(end_node[0])
    # slack connection node will be given an extra flow condition
    nodes_list.remove(slack_connection_node)
    nodes_list.remove('s')
    
    for node in nodes_list:
        ingoing_edges = get_ingoing_edges(df,node)
        outgoing_edges = get_outgoing_edges(df,node)
 
        sum_Q_out = sum(Q[in_edge][n-1,:] for in_edge in ingoing_edges)
        sum_Q_in = sum(Q[out_edge][0,:] for out_edge in outgoing_edges)
        
        g += [(sum_Q_in - sum_Q_out).reshape((-1,1))]
        lbg += [0.,] * m
        ubg += [0.,] * m
    
    #### p_node = p_pipe
    # remove compressor node
    for node in list_of_compressors:
        nodes_list.remove(node)
    nodes_list.append(slack_connection_node)
    
    for node in nodes_list: 
        ingoing_edges = get_ingoing_edges(df, node)
        outgoing_edges = get_outgoing_edges(df, node)

        for edge_in in ingoing_edges:
            for edge_out in outgoing_edges:
            # Only for edges which are last edge (thus slack edge)
                if edge_out != number_of_edges_without_slack_edge:
                    g += [(P[edge_in][n-1,:] - P[edge_out][0,:]).reshape((-1,1))]
                    lbg += [0.] * m
                    ubg += [0.] * m
    print("Condition 2 is set.")
    
    ####################
    #### Condition 3 ###
    # Properties at compressor station
    
    # list containing all compressor configurations
    for j, com in enumerate(list_of_compressors):
        ingoing_edge = get_ingoing_edges(df, com) 
        outgoing_edge = get_outgoing_edges(df, com)

        # In our model there is one ingoing and one outgoing edge for every compressors
        # Angst vor zwei unterschiedlichen zusammenlaufenden Drücken?
        if len(ingoing_edge) == 1 and len(outgoing_edge) == 1: 
            ingoing_edge = ingoing_edge[0]
            outgoing_edge = outgoing_edge[0]

            g += [(P[outgoing_edge][0,:] - P[ingoing_edge][n-1,:] - (u[:,j]).reshape((1,-1))).reshape((-1,1))]
            lbg += [0.] * m
            ubg += [0.] * m

            # no pressure increase condition
            g += [u[:,j]*beta[:,j] - u[:,j]] 
            lbg += [0.] * m
            ubg += [0.] * m
            
    print("Condition 3 is set.")
    
    ####################
    #### Condition 4 ###
    # Properties at slack connection node
    
    list_outgoing_edges = get_outgoing_edges(df,slack_connection_node)
    list_ingoing_edges = get_ingoing_edges(df,slack_connection_node)
    
    # (!) Assumption: we assume that there is only one further outgoing edge besides
    # the slack connection edge
    for j in list_outgoing_edges:
        if j != number_of_edges_without_slack_edge: # not the last edge    
            filtered_slack_connection_node_out_edges = j 

    print("What is slack connection node: " + str(filtered_slack_connection_node_out_edges))
    
    sum_of_Q = sum(Q[edge][n-1,:] for edge in list_ingoing_edges)
    
    g += [(sum_of_Q - Q[filtered_slack_connection_node_out_edges][0,:] - eps[:].reshape((1,-1))).reshape((-1,1))]
    lbg += [0.] * m
    ubg += [0.] * m
    print("Condition 4 is set.")
    
    #############################
    #### Optional Condition 6 ###
    # Additional constraints that couple over time
    if additional_constraints == True:
        # Switch has to keep its state for at least time M1 from off to on
        c = [list(i) for i in itertools.product([0, 1], repeat = number_of_compressors)]
        
        for j in range(number_of_compressors):
            for k in range(m - math.floor(M1/dt)): # (m-6)
                rside = sum(c[con][j]*math.floor(M1/dt)*(-beta[k,con] + beta[k+1,con])
                    for con in range(number_of_configs))
                lside = sum(c[con][j]*beta[e,con]
                    for con in range(number_of_configs) 
                    for e in range(k+1, k + math.floor(M1/dt))) 
                g += [lside - rside]
                lbg += [0]
                ubg += [+cas.inf]
        
        # Switch has to keep its state for at least time M2 from on to off
        for j in range(number_of_compressors):
            for k in range(m - math.floor(M2/dt)): # (m-6)
                rside = sum(c[con][j]*math.floor(M2/dt)*(beta[k,con] - beta[k+1,con])
                                                    for con in range(number_of_configs))
                lside = sum(1 - sum(c[con][j]*beta[e,con] 
                        for con in range(number_of_configs)) 
                        for e in range(k+1, k + math.floor(M2/dt)))
                g += [lside - rside]
                lbg += [0]
                ubg += [+cas.inf]
        
        print("Additional conditions are set.")
    
    ###########################
    #### Objective function ###
    
    J = 0.5 * sum(u[t,j]**2 for t in range(m) for j in range(number_of_compressors))
    print("Objective function set.")
    
    # Create NLP dictionary
    parameters = [m, n, number_of_compressors, number_of_configs, number_of_edges_without_slack_edge, starting_edges, end_edge]
    nlp = {}
    nlp['f'] = J
    nlp['x'] = cas.vertcat(*w)
    nlp['g'] = cas.vertcat(*g)

    return parameters, nlp, lbw, ubw, lbg, ubg, w0, discrete


def extract_solution(sol, parameters):
    """
    Function for setting up the NLP
    input: sol, parameters
    output: beta, u, P, Q
    """
    m, n, number_of_compressors, number_of_configs, number_of_edges, starting_edges, end_edge = parameters
    offset = 0
    u = np.array(cas.reshape(sol['x'][offset:offset + number_of_compressors * m],
        m, number_of_compressors))
    offset += number_of_compressors * m
    beta = np.array(cas.reshape(sol['x'][offset:offset + number_of_compressors * m],
        m, number_of_compressors))
    offset += number_of_compressors * m
    P, Q = [], []
    for i in range(number_of_edges):
        P += [np.array(cas.reshape(sol['x'][offset:offset + m * n], n, m))]
        offset += m * n
        Q += [np.array(cas.reshape(sol['x'][offset:offset + m * n], n, m))]
        offset += m * n
    
    return beta, u, P, Q

def plot_solution(beta, u, P, Q, parameters, final_directory):
    """
    Function for plotting the solution 
    input: alpha, u, P, Q
    """
    m, n, number_of_compressors, number_of_configs, number_of_edges, starting_edges, end_edge = parameters
    t = np.arange(0,m)
    
    # plot beta
    plt.figure(1).clear()
    fig, axes  = plt.subplots(number_of_compressors, 1, num=2)
    axes = axes.reshape((-1,)) if number_of_compressors > 1 else [axes]
    for i in range(number_of_compressors):
        axes[i].step(t, beta[:,i])
        axes[i].set_xlabel(r'time step $t$')
        axes[i].set_ylabel(r'Compressor {}: on/off'.format(i))
    plt.savefig(final_directory + 'bonmin_control_beta.png')
    
    # plot u
    plt.figure(2).clear()
    fig, axes  = plt.subplots(number_of_compressors, 1, num=2)
    axes = axes.reshape((-1,)) if number_of_compressors > 1 else [axes]
    for i in range(number_of_compressors):
        axes[i].step(t, u[:,i])
        axes[i].set_xlabel(r'time step $t$')
        axes[i].set_ylabel(r'control $u_{}$'.format(i))
    plt.savefig(final_directory + 'bonmin_control_u.png')
    
    # plot inflow at first node
    plt.figure(4).clear()
    fig, axes = plt.subplots(num = 4)
    axes.plot(t, Q[starting_edges[0]][0,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'mass flow (kg/s), first node')
    plt.savefig(final_directory + 'bonmin_flow_firstNode.png')
    
    # plot pressure at first node
    plt.figure(5).clear()
    fig, axes = plt.subplots(num = 5)
    axes.plot(t, P[starting_edges[0]][0,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'pressure (bar), first node')
    plt.savefig(final_directory + 'bonmin_pressure_firstNode.png')
    
    # plot pressure at last node
    plt.figure(6).clear()
    fig, axes = plt.subplots(num = 6)
    axes.plot(t, P[end_edge][n-1,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'pressure (bar), last node')
    axes.set_ylim(59,61)
    plt.savefig(final_directory + 'bonmin_pressure_lastNode.png')
    
    # plot inflow at first node
    plt.figure(7).clear()
    fig, axes = plt.subplots(num = 7)
    axes.plot(t, Q[end_edge][0,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'mass flow (kg/s), last node')
    plt.savefig(final_directory + 'bonmin_flow_lastNode.png')
    
    
if __name__ == '__main__':
     P_initialnode = np.loadtxt(folder + 'P_initialnode.dat')
     Q_lastnode = np.loadtxt(folder + 'Q_lastnode.dat')
     P_time0 = np.loadtxt(folder + 'P_time0.dat')
     Q_time0 = np.loadtxt(folder + 'Q_time0.dat')
     Edgesfile = folder + 'Edges.txt'
     
     # Which eps file
     if folder == 'Example_Advanced/':
         eps_file = 'eps_file.mat'
         mat_file = scipy.io.loadmat(folder + eps_file)
         eps = mat_file["eps"]
         eps[0,0] = round(eps[0,0],4) # to make things a little simpler 
     elif folder == 'Example_Simple/':
         eps_file = 'eps_file.dat'
         eps = np.loadtxt(folder + eps_file)

     current_directory = os.getcwd()
     final_directory = os.path.join(current_directory, folder + '/bonmin_noPOC/')
     if not os.path.exists(final_directory):
         os.makedirs(final_directory)
     parameters, nlp, lbw, ubw, lbg, ubg, w0, discrete = gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_lastnode, eps, Edgesfile)
    
     # Solving the problem with BONMIN solver    
     options = {'bonmin': {'time_limit': 1800}, 'discrete': discrete} # 1800 sec = 30 min
     t0_start = process_time()
     solver = cas.nlpsol('nlp_solver', 'bonmin', nlp, options);
     sol = solver(x0 = w0, lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
     beta, u, P, Q = extract_solution(sol, parameters)
     t0_stop = process_time()
     elapsed_time = t0_stop - t0_start
     print("Time elapsed during execution of bonmin :" + str(elapsed_time))
     plot_solution(beta, u, P, Q, parameters, final_directory)
     