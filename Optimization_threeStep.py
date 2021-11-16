#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25
Compressor optimization with partial outer convexification
using three-step approach.
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
import gurobipy as gp
from gurobipy import GRB
import os

# Set folder name, provide like this: 'Example/'
folder = 'Example_Advanced/'

###############################
#### Additional Constraints ###
# Additional constrains that couple over time: True - on, False - off
# add1: compressor can at least switch e.g. 5 times,
# add2: compressor can switch after certain time steps

additional_constraints = True
add = 'add1'
if additional_constraints and add == 'add2':
    M1, M2 = 1/20, 1/20
    
# Constraints on P and on Q
lbP = 0 # there is no negative pressure
ubP = 100 
lbQ = -1000
ubQ = +1000

# Read Config Parameters in Configs.txt
config = configparser.ConfigParser()
config.read(folder + 'Configs.txt')

length_of_pipe = int(config['configs']['LengthPipe']) # universal length for all pipes (in m)
time_in_total = int(config['configs']['TimeInTotal']) # time (in sec)
Lambda = float(config['configs']['Lambda']) # here: lambda is fix
D = int(config['configs']['Diameter']) # in m
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
    # remove slack node
    end_nodes.remove('s') 
    
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
                # break, because starting node is only connected to one edge
                break  
    
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
    # make list unique
    unique_list_of_compressors = list(set(list_of_compressors)) 
    
    return unique_list_of_compressors


def get_slack_connection_node(df):
    """
    Function for extracting the node which is connected to slack bus in Edges.txt
    assumption: only one node is connected to slack bus
    input: dataframe
    output: int
    """
    for i in range(df.shape[0]):
        if df.iloc[i,2] == 's':
            return df.iloc[i,1]


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
            # there is only one edge
            return df.iloc[edge,0]


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


def gasnetwork_nlp(P_time0, Q_time0, eps, Edgesfile):
    """
    Function for setting up the NLP
    input: eps, P_time0, Q_time0
           Edgesfile as string file -> ('Edges.txt')
    output: nlp, lbw, ubw, lbg, ubg, w0
    """
    n = np.shape(P_time0)[1]
    print("This is the number of space steps: " + str(n))
    m = np.shape(eps)[0] 
    print("This is the number of time steps: " + str(m))
    dx = length_of_pipe/n # in (m)
    dt = time_in_total/m  # in (s) 
    
    # Is CFL fulfilled?
    if (dt*a_square > dx):
        raise Exception("CFL is not fulfilled.")
    # Are dimensions uniformy?
    if (np.shape(P_time0) != np.shape(Q_time0)):
        raise Exception("Mismatch of dimension of Q_time0, P_time0.")
    
    df = pd.read_csv(Edgesfile, header = None)
    df = df.iloc[1:, :]
    df = df.reset_index(drop = True)
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
    P, Q, u = [], [], []
    # alpha does not need vector initialization
    
    ################################
    ### Set initial conditions #####
    # alpha
    alpha = cas.MX.sym('alpha', m, number_of_configs)
    w += [cas.reshape(alpha, -1, 1)] # reshape columnwise 
    w0 += [.5] * m * number_of_configs
    lbw += [0.] * m * number_of_configs
    ubw += [1.] * m * number_of_configs 
    
    # u
    u = cas.MX.sym('u', m, number_of_compressors)
    w += [cas.reshape(u, -1, 1)]
    w0 += [0.] * m * number_of_compressors  
    lbw += [0.] * m * number_of_compressors
    ubw += [+cas.inf] * m * number_of_compressors
    
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
            # For number_of_edges_without_slack_edge there is no P,Q
                if edge_out != number_of_edges_without_slack_edge:
                    g += [(P[edge_in][n-1,:] - P[edge_out][0,:]).reshape((-1,1))]
                    lbg += [0.] * m
                    ubg += [0.] * m
    print("Condition 2 is set.")
    
    ####################
    #### Condition 3 ###
    # Properties at compressor station
    
    # SOS1 constraint
    g += [cas.mtimes(alpha, cas.DM.ones(number_of_configs))]
    lbg += [1.] * (m)
    ubg += [1.] * (m)
    
    # list containing all compressor configurations
    c = [list(i) for i in itertools.product([0, 1], repeat = number_of_compressors)]
    
    for j, com in enumerate(list_of_compressors):
        ingoing_edge = get_ingoing_edges(df, com) 
        outgoing_edge = get_outgoing_edges(df, com)

        # In our model there is one ingoing and one outgoing edge for every compressors
        if len(ingoing_edge) == 1 and len(outgoing_edge) == 1: 
            ingoing_edge = ingoing_edge[0] 
            outgoing_edge = outgoing_edge[0]

            sum_com = sum(c[s][j]*alpha[:,s]*u[:,j] for s in range(number_of_configs))
            # no pressure increase condition
            g += [u[:,j] - sum_com]
            lbg += [0.] * m
            ubg += [0.] * m
            
            g += [(P[outgoing_edge][0,:] - P[ingoing_edge][n-1,:] - u[:,j].reshape((1,-1))).reshape((-1,1))]
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
        if j != number_of_edges_without_slack_edge: # not sink "edge"
            filtered_slack_connection_node_out_edges = j

    print("What is slack connection node: " + str(filtered_slack_connection_node_out_edges))
    sum_of_Q = sum(Q[edge][n-1,:] for edge in list_ingoing_edges)
    
    g += [(sum_of_Q - Q[filtered_slack_connection_node_out_edges][0,:] - eps[:].reshape((1,-1))).reshape((-1,1))]
    lbg += [0.] * m
    ubg += [0.] * m
    print("Condition 4 is set.")
        
    ###########################
    #### Objective function ###
    J = 0.5 * sum(u[t,j]**2 for t in range(m) for j in range(number_of_compressors))
    print("Objective function is set.")

    # Create NLP dictionary
    parameters = [dt, m, n, number_of_compressors, number_of_configs, number_of_edges_without_slack_edge, starting_edges, end_edge]
    nlp = {}
    nlp['f'] = J
    nlp['x'] = cas.vertcat(*w)
    nlp['g'] = cas.vertcat(*g)

    return parameters, nlp, lbw, ubw, lbg, ubg, w0


def CIAP_SumUpRounding(alpha):
    """
    Sum Up Rounding for SOS1
    input: alpha
    output: p
    """
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


def CIAP_MINLP(alpha, parameters):
    """
    Solving CIAP with additional constraints with gurobipy (MILP)
    input: alpha, parameters
    output: beta
    """    
    dt, m, n, number_of_compressors, number_of_configs, number_of_edges, starting_edges, end_edge = parameters
    model = gp.Model("CIAP")
    eps = model.addVar(vtype = 'C', name = 'eps')
    delta0 = model.addVars(number_of_configs, vtype = 'C', name = 'delta0')
    beta = model.addVars(m, number_of_configs, vtype = 'B', name = 'beta')
    z = model.addVars(m,1,vtype = 'B',name='z')
    phi = model.addVars(m,1,vtype = 'B',name='phi')
    
    # list containing all compressor configurations
    c = [list(i) for i in itertools.product([0, 1], repeat = number_of_compressors)]
    
    model.setObjective(eps, GRB.MINIMIZE)

    # Deviation Constraint
    for con in range(number_of_configs):
        for k in range(m):
            model.addConstr(delta0[con] + dt*sum(alpha[t,con] - beta[t,con]
                                          for t in range(k-1)) <= eps)
            model.addConstr(delta0[con] + dt*sum(alpha[t,con] - beta[t,con]
                                          for t in range(k-1)) >= (-1.0) * eps)

    # Additional constraints that couple over time
    if add == 'add1':
        # Compressor can switch max_switch_number amount of time
        sum_of_diff = 0
        max_switch_number = 10
        for com in range(number_of_compressors):
            for t in range(1,m):
                for con in range(number_of_configs):
                    model.addConstr(phi[t,0] == c[con][com]*beta[t,con] - c[con][com]*beta[t-1,con])
                    model.addGenConstrAbs(z[t,0], phi[t,0])
                    sum_of_diff = sum_of_diff + z[t,0]
        model.addConstr(sum_of_diff <= max_switch_number)
        print('Additional Constraints 1 is set.')
         
    elif add == 'add2':
        for j in range(number_of_compressors):
            for k in range(m - math.floor(M1/dt)):
                # Compressor can switch from off to on after M1 time has passed
                rside = sum(c[con][j]*math.floor(M1/dt)*(-beta[k,con] + beta[k+1,con])
                        for con in range(number_of_configs))
                
                lside = sum(c[con][j]*beta[e,con]
                        for con in range(number_of_configs) 
                        for e in range(k+1, k + math.floor(M1/dt))) 
                model.addConstr(lside - rside >= 0)
                
            for k in range(m - math.floor(M2/dt)):
                # Compressor can switch from on to off after M2 time has passed
                rside = sum(c[con][j]*math.floor(M2/dt)*(beta[k,con] - beta[k+1,con])
                                                       for con in range(number_of_configs))
                lside = sum(1 - sum(c[con][j]*beta[e,con] 
                            for con in range(number_of_configs)) 
                            for e in range(k+1, k + math.floor(M2/dt)))
                model.addConstr(lside - rside >= 0)
        print('Additional Constraints 2 is set.')

    # SOS-Type1 constraint
    for t in range(0, m):
        model.addConstr(sum(beta[t,con] for con in range(number_of_configs)) == 1)
    
    print("SOS1 is set.")
    
    model.Params.TimeLimit = 5*60 # maximum running time: 5 minutes
    model.optimize()
    eps_obj = model.objVal
    
    # Save tupledict variables in a matrix
    sol_beta = np.zeros((m,number_of_configs))
    tupledict_beta = model.getAttr('x', beta)
    for t in range(m):
        for j in range(number_of_configs):
            sol_beta[t,j] = tupledict_beta[t,j]
    
    # Save delta0 variables in a vector
    delta0_sol = np.zeros((number_of_configs, 1))
    tupledict_delta0 = model.getAttr('x', delta0)
    for j in range(number_of_configs):
        delta0_sol[j] = tupledict_delta0[j]
    
    return eps_obj, delta0_sol, sol_beta

    
def plot_solution(alpha, u, P, Q, parameters, final_directory):
    """
    Function for plotting the solution 
    input: alpha, u, P, Q
    """
    dt, m, n, number_of_compressors, number_of_configs, number_of_edges, starting_edges, end_edge = parameters
    t = np.arange(0,m)
    
    c = [list(i) for i in itertools.product([0, 1], repeat = number_of_compressors)]
    
    # plot u
    plt.figure(1).clear()
    fig, axes  = plt.subplots(number_of_compressors, 1, num=1)
    axes = axes.reshape((-1,)) if number_of_compressors > 1 else [axes]
    for i in range(number_of_compressors):
        axes[i].step(t, u[:,i])
        axes[i].set_xlabel(r'time step $t$')
        axes[i].set_ylabel(r'control $u_{}$'.format(i))
    plt.savefig(final_directory + 'POCControl_u.png')
    
    # plot compressor on/off
    plt.figure(2).clear()
    # vector provides which configuration is on for compressors
    configurations_on = []
    for time_step in range(m):
        case = False
        j = 0
        while case == False:
            if alpha[time_step, j] == 1:
                configurations_on.append(c[j])
                case = True
            j = j + 1
    configurations_on_vec = np.array(configurations_on)       
    fig, axes  = plt.subplots(number_of_compressors, 1, num=2)
    axes = axes.reshape((-1,)) if number_of_compressors > 1 else [axes]
    for i in range(number_of_compressors):
         axes[i].step(t, configurations_on_vec[:,i])
         axes[i].set_xlabel(r'time step $t$')
         axes[i].set_ylabel(r'Compressor {}: on/off'.format(i))
    plt.savefig(final_directory + 'POCConf.png')
    
    # plot inflow at first node
    plt.figure(4).clear()
    fig, axes = plt.subplots(num = 4)
    axes.plot(t, Q[starting_edges[0]][0,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'mass flow (kg/s) at first node')
    plt.savefig(final_directory + 'POCflow_firstNode.png')
    
    # plot pressure at first node
    plt.figure(5).clear()
    fig, axes = plt.subplots(num = 5)
    axes.plot(t, P[starting_edges[0]][0,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'pressure (bar) at first node')
    plt.savefig(final_directory + 'POCpressure_firstNode.png')
    
    # plot outflow at last node
    plt.figure(6).clear()
    fig, axes = plt.subplots(num = 6)
    axes.plot(t, Q[end_edge][0,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'mass flow (kg/s) at last node')
    plt.savefig(final_directory + 'POCflow_lastNode.png')
    
    # plot pressure at last node
    plt.figure(7).clear()
    fig, axes = plt.subplots(num = 7)
    axes.plot(t, P[end_edge][0,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'pressure (bar) at last node')
    axes.set_ylim(59,61)
    plt.savefig(final_directory + 'POCpressure_lastNode.png')
    

def extract_solution(sol, parameters):
    """
    Function for Extracting Solution of the POC NLP
    input: sol, parameters
    output: alpha, u, P, Q
    """
    dt, m, n, number_of_compressors, number_of_configs, number_of_edges, starting_edges, end_edge = parameters
    offset = 0
    alpha = np.array(cas.reshape(sol['x'][offset:offset + m * number_of_configs],
        m, number_of_configs))
    offset += m * number_of_configs
    u = np.array(cas.reshape(sol['x'][offset:offset + number_of_compressors * m],
        m, number_of_compressors))
    offset += number_of_compressors * m
    P, Q = [], []
    for i in range(number_of_edges):
        P += [np.array(cas.reshape(sol['x'][offset:offset + m * n], n, m))]
        offset += m * n
        Q += [np.array(cas.reshape(sol['x'][offset:offset + m * n], n, m))]
        offset += m * n
    
    return alpha, u, P, Q
    
if __name__ == '__main__':
     P_time0 = np.loadtxt(folder + 'P_time0.dat')
     Q_time0 = np.loadtxt(folder + 'Q_time0.dat')
     Edgesfile = folder + 'Edges.txt'
     
     # determine which eps file is taken
     if folder == 'Example_Advanced/':
         eps_file = 'eps_file.mat'
         mat_file = scipy.io.loadmat(folder + eps_file)
         eps = mat_file["eps"]
         # to match with initial data
         eps[0,0] = round(eps[0,0],4) 
     elif folder == 'Example_Simple/':
         eps_file = 'eps_file.dat'
         eps = np.loadtxt(folder + eps_file)

     parameters, nlp, lbw, ubw, lbg, ubg, w0 = gasnetwork_nlp(P_time0, Q_time0, eps, Edgesfile)
    
     # Solving the problem with IPOPT solve and library 'ma27', 
     # since it has better performance for large nlp problems
     options = {'ipopt': {'tol': 1e-8, 'max_iter': 1100, 'linear_solver': 'ma27'}}

     # Measure execution time of algorithm
     t0_start = process_time()
     
     solver = cas.nlpsol('solver', 'ipopt', nlp, options)
     sol = solver(x0 = w0, lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
     alpha, u, P, Q = extract_solution(sol, parameters)
     
     t0_stop_Step1 = process_time()
     
     if additional_constraints == True:
         eps_obj,delta0, alpha2 = CIAP_MINLP(alpha, parameters)
     else:
         alpha2 = CIAP_SumUpRounding(alpha)
         alpha2.reshape(-1, 1)

     for i, v in enumerate(np.reshape(alpha2, -1, order='F')):
         lbw[i] = v
         ubw[i] = v

     t0_stop_Step2 = process_time()

     sol2 = solver(x0 = sol['x'], lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)
     alpha3, u2, P2, Q2 = extract_solution(sol2, parameters)
     
     t0_stop_Step3 = process_time()
     
     # Computing Execution Time: floating number expressed in seconds
     Step1_time = t0_stop_Step1 - t0_start
     Step2_time = t0_stop_Step2 - t0_stop_Step1
     Step3_time = t0_stop_Step3 - t0_stop_Step2
     time_total = t0_stop_Step3 - t0_start
     
     print("Time elapsed, Step 1: " + str(Step1_time) + "\n")
     print("Time elapsed, Step 2: " + str(Step2_time) + "\n")
     print("Time elapsed, Step 3: " + str(Step3_time) + "\n")
     print("Total time elapsed during execution: " + str(time_total))
     print("Folder: " + str(folder) + " was used with eps: " + str(eps_file))
     
     # Plot solution
     current_directory = os.getcwd()
     final_directory = os.path.join(current_directory, folder + '3step_POC_AC2/')
     if not os.path.exists(final_directory):
         os.makedirs(final_directory)
     plot_solution(alpha3, u2, P2, Q2, parameters, final_directory)