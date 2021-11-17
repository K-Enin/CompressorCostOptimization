# Compressor Optimization with Partial Outer Convexification (POC)

## General Information

In this code we consider different Gas-to-Power networks.
The aim of this code is to solve an optimization problem, such that all constraints (gas-coupling conditions, Euler Equation, slack & compressor condition) are satisfied and the objective function which is the sum of all pressure differences at the compressor nodes is minimized. \
Instead of Euler Equation we use Weymouth Equation which provides similar results and is better to control in the terms of the CFL condition. \
We discretize the Weymouth Equation with Simple Upwind method. 

## Folders

This folder contains four different folders: Example_Advanced, Example_Simple, powergrid, Simulation. \
The 'Simulation'-folder contains two scripts which compare the Weymouth and Euler Equation with different discretization methods. Furthermore it contains a small Example of Sum Up Rounding.
The matlab scripts provided in 'power grid' calculate the mass flux which is needed for the power network.
Example_Simple and Example_Advanced contain initial data and parameters which are used for optimizing the gas-network model.

## Scripts

This folder contains the following python scripts:
Optimization_NLP.py
OptimizationBonminNoPOC.py
OptimizationPOC_threeStep.py
OptimizationBonminPOC.py
generateInitialData.py

The script 'Optimization_threeStep.py' uses the three step approach for compressor cost optimization.
'OptimizationBonminNoPOC.py' uses the direct solver bonmin to obtain a solution for compressor cost optimization.'OptimizationBonminPOC.py' uses again bonmin, but the optimization problem is reformulated using partial outer convexification. Additionally 'Optimization_NLP.py' is an NLP Problem which was used for testing purposes.
The last script 'generateInitialData.py' was used to generate data in 'Example_Simple' and 'Example_Advanced'.

## Run the Code

Each example folder (Example_Simple and Example_Advanced) contains the following files: 
eps_file.dat, P_time0.dat, Q_time0.dat, Edges.txt, Configs.txt and an illustration of the network. 
- eps_file.dat contains a 1xm vector of the mass flux taken out of the gas network to the slack bus for power conversion. 
It is set beforehand using generateInitialData.py or calculated using the scripts in power grid. 
- P_time0.dat and Q_time0.dat contain all initial pressure and flux values in time t=0 for every pipe section in every pipe (except the pipe which is connected to the slack bus, since its simulation is not important).  
- Edges.txt contain all the dependencies between nodes. 
- Configs.txt contain several config parameters which characterize our gas network model. 

For each type of Optimization Problem an image folder is generated in the corresponding Example folder.

## Remark
When solving the optimization problem with IPOPT we use library "ma27", 
which has to be installed externally and is not initially included in CasAdi. 
According to CasAdi Google Group library 'ma27' has a better performance for large nlp problems.
