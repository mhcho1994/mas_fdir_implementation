"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - 3D Project
            - Implementation of " Collaborative Fault-Identification &
              Reconstruction in Multi-Agent Systems" by Khan et al.
            - Algorithm uses inter-agent distances to reconstruct a sparse
              vector of agents where the nonzero elements are exactly the faulty
              agents, with the elements being the attack vectors. The algorithm
              does not assume any anchors exist so entire network can be evaluated
              for faults.
            - Uses SCP to convexify the nonconvex problem.
            - Uses ADMM to split the convex problem into smaller problems that can
              be solved parallelly by each agent.
"""

###     Imports             - Public Libraries
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm import tqdm

###     Imports             - User-Defined Files
from generic_agent import GenericAgent as Agent
from iam_models import distance



###     Initializations     - Scalars
dim             =   3   # 2 or 3
num_agents      =   7
num_faulty      =   1   # must be << num_agents for sparse error assumption
n_scp           =   10  # Number of SCP iterations
n_admm          =   10  # Number of ADMM iterations
n_iter          =   n_admm * n_scp
show_prob1      =   False
show_prob2      =   False



###     Initializations     - Agents
# 5 agents making up square pyramid, 1 agents at center of square
agents      =   [None] * num_agents
d           =   10      # square side length
agents[0]   =   Agent(agent_id= 0,
                      init_position= np.array([[0, 0, d]]).T) #np.array([[0, 0, d]]).T)
agents[1]   =   Agent(agent_id= 1,
                      init_position= np.array([[d/3, d/2, d/4]]).T) #np.array([[d/2, d/2, 0]]).T)
agents[2]   =   Agent(agent_id= 2,
                      init_position= np.array([[d/5, -d/2, 0]]).T) #np.array([[d/2, -d/2, 0]]).T)
agents[3]   =   Agent(agent_id= 3,
                      init_position= np.array([[-d/3, -d, d/5]]).T) #np.array([[-d/2, -d/2, 0]]).T)
agents[4]   =   Agent(agent_id= 4,
                      init_position= np.array([[-d/4, d/2, -d/4]]).T) #np.array([[-d/2, d/2, 0]]).T)
agents[5]   =   Agent(agent_id= 5,
                      init_position= np.array([[0, 0, 0]]).T) #np.array([[0, 0, 0]]).T)
agents[6]   =   Agent(agent_id= 6,
                      init_position= np.array([[0, 0, -2*d]]).T) #np.array([[0, 0, -d]]).T)


# Add error vector
faulty_id   =   1 # np.random.randint(0, high=num_agents)
fault_vec   =   np.array([[0.0, 0, 0]]).T # 0.5*np.random.rand(dim, 1) # 
agents[faulty_id].faulty = True
agents[faulty_id].error_vector = fault_vec

x_true = []
for id, agent in enumerate(agents):
    x_true.append(agent.error_vector)


# Set Neighbors
edges                   = [[0,1], [0,2], [0,3],
                           [0,4], [0,5], [1,2],
                           [1,4], [1,5], [1,6],
                           [2,3], [2,5], [2,6],
                           [3,4], [3,5], [3,6], 
                           [4,5], [4,6], [5,6],

                           [1,0], [2,0], [3,0],
                           [4,0], [5,0], [2,1],
                           [4,1], [5,1], [6,1],
                           [3,2], [5,2], [6,2],
                           [4,3], [5,3], [6,3],
                           [5,4], [6,4], [6,5]] # these edges are directed
for agent_id, agent in enumerate(agents):
    # Neighbor List
    nbr_list        =   []
    edge_list       =   []
    
    for edge_ind, edge in enumerate(edges):
        if (agent_id) == edge[0]:
            nbr_list.append(edge[1])
            edge_list.append(edge_ind)
    
    agent.set_neighbors(nbr_list)
    agent.set_edge_indices(edge_list)



###     Useful Functions
# Measurement function Phi
def measurements(p, x_hat):
    measurements = []

    for edge in edges:
        dist = distance((p[edge[0]] + x_hat[edge[0]]), (p[edge[1]] + x_hat[edge[1]]))
        measurements.append(dist)

    return measurements

# Finds row of R
def get_Jacobian_row(edge_ind, p, x):
    edge = edges[edge_ind]
    agent1_id = edge[0]
    agent2_id = edge[1]
    disp    = ((p[edge[1]] + x[edge[1]]) - 
               (p[edge[0]] + x[edge[0]]))
    R_k = np.zeros((1, dim*num_agents))

    R_k[:, dim*agent2_id:dim*(agent2_id + 1)] = disp.T
    R_k[:, dim*agent1_id:dim*(agent1_id + 1)] = -disp.T

    return R_k

# Computes whole R matrix
def get_Jacobian_matrix(p, x):
    R = []

    for edge_ind, edge in enumerate(edges):
        R.append(get_Jacobian_row(edge_ind, p, x))
    
    return R



###     Initializations     - Measurements and Positions
x_star = [np.zeros((dim, 1)) for i in range(num_agents)]                    # Equivalent to last element in x_history (below)
x_history = [np.zeros((dim, (n_iter))) for i in range(num_agents)]          # Value of x at each iteration of algorithm
x_norm_history = [np.zeros((1, (n_iter))) for i in range(num_agents)]       # Norm of difference between x_history and x_true
p_est = [agents[i].get_estimated_pos() for i in range(num_agents)]          # Will be updated as algorithm loops and err vector is reconstructed
p_hat = deepcopy(p_est)                                                     # CONSTANT: Reported positions of agents
p_true = [agents[i].get_true_pos() for i in range(num_agents)]              # CONSTANT: True pos
y = measurements(p_true, x_star)                                            # CONSTANT: Phi(p_hat + x_hat), true interagent measurement



###      Initializations    - Optimization Parameters
rho = 1.0
total_iterations = np.arange((n_iter))
for agent_id, agent in enumerate(agents):
    num_edges       = len(agent.get_edge_indices())
    num_neighbors   = len(agent.get_neighbors())

    # CVX variables
    agent.init_x_cp(cp.Variable((dim, 1)))
    agent.init_w_cp(cp.Variable((dim, 1)), agent.get_neighbors())

    # Parameters
    agent.init_x_bar(np.zeros((dim, 1)))
    agent.init_lam(np.zeros((1, 1)), agent.get_edge_indices())
    agent.init_mu(np.zeros((dim, 1)), agent.get_neighbors())
    agent.init_x_star(np.zeros((dim, 1)), agent.get_neighbors()) # own err is last elem
    agent.init_w(np.zeros((dim, 1)), agent.get_neighbors())


###     Initializations     - List Parameters
print("\n~ ~ ~ ~ PARAMETERS ~ ~ ~ ~")
print("rho:", rho)
print("Number of agents:", num_agents)
print("Faulty Agent ID:", faulty_id)
print("Faulty Agent Vector:", fault_vec.T)


###     Looping             - SCP Outer Loop
print("\nStarting Loop")
for outer_i in tqdm(range(n_scp), desc="SCP Loop", leave=True):
    new_measurement = measurements(p_hat, x_star)
    z       =   [(y[i] - meas) for i, meas in enumerate(new_measurement)]
    R       =   get_Jacobian_matrix(p_hat, x_star)

    for agent in agents:
        agent.init_w(np.zeros((dim, 1)), agent.get_neighbors())


    ###     Looping             - ADMM Inner Loop
    for inner_i in tqdm(range(n_admm), desc="ADMM Loop", leave=False):

        ##      Minimization        - Primal Variable 1
        for agent_id, agent in enumerate(agents):
            objective = cp.norm(agent.x_star[agent_id] + agent.x_cp)
            
            # Summation for c() constraint
            for _, edge_ind in enumerate(agent.get_edge_indices()): 
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_cp - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w[agent_id]
                
                objective += ((rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[edge_ind].T @ (constr_c))
            
            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()): 
                constr_d = agent.x_cp - agent.w[nbr_id]
                objective += ((rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob1 = cp.Problem(cp.Minimize(objective), [])
            prob1.solve(verbose=show_prob1)
            if prob1.status != cp.OPTIMAL:
                print("\nERROR Problem 1: Optimization problem not solved @ (%d, %d, %d)" % (inner_i, outer_i, agent_id))

            agent.x_bar = deepcopy(np.array(agent.x_cp.value).reshape((-1, 1)))
            new_x = deepcopy(agent.x_bar.flatten()) + x_star[agent_id].flatten()

            x_history[agent_id][:, inner_i + outer_i*n_scp] = new_x.flatten()
            x_norm_history[agent_id][:, inner_i + outer_i*n_scp] = np.linalg.norm(new_x.flatten() - x_true[agent_id].flatten())

        ##      Minimization        - Thresholding Parameter
        # TODO: Implement
        # Used for identifying faults, not pressing issue

        ##      Minimization        - Primal Variable 2
        for agent_id, agent in enumerate(agents):
            objective = cp.norm(agent.x_star[agent_id] + agent.x_bar)

            # Summation for c() constraint
            for edge_ind in agent.get_edge_indices(): 
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c = constr_c + R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w_cp[agent_id]
                
                objective += ((rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[edge_ind].T @ (constr_c))
            
            # Summation for d() constraint
            for nbr_id in agent.get_neighbors():
                constr_d = agent.x_bar - agent.w_cp[nbr_id]
                objective += ((rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob2 = cp.Problem(cp.Minimize(objective), [])
            prob2.solve(verbose=show_prob2)
            if prob2.status != cp.OPTIMAL:
                print("\nERROR Problem 2: Optimization problem not solved @ (%d, %d, %d)" % (inner_i, outer_i, agent_id))

            for _, nbr_id in enumerate(agent.get_neighbors()):
                agent.w[nbr_id] = deepcopy(np.array(agent.w_cp[nbr_id].value).reshape((-1, 1)))


        ##      Multipliers         - Update Lagrangian Multipliers of Minimization Problem
        for agent_id, agent in enumerate(agents):
            
            # Summation for c() constraint
            for _, edge_ind in enumerate(agent.get_edge_indices()):
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w[agent_id]
                
                agent.lam[edge_ind] = deepcopy(agent.lam[edge_ind] + rho * constr_c)

            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()):
                constr_d = agent.x_bar - agent.w[nbr_id]
                agent.mu[nbr_id] = deepcopy(agent.mu[nbr_id] + rho * constr_d)

    ###     END Looping         - ADMM Inner Loop
    
    # Update Error Vectors after ADMM subroutine
    for agent_id, agent in enumerate(agents): 
        for list_ind, nbr_id in enumerate(agent.get_neighbors()):
            agent.x_star[nbr_id] = agent.x_star[nbr_id] + agents[nbr_id].x_bar
        
        agent.x_star[agent_id] = agent.x_star[agent_id] + agent.x_bar
        x_star[agent_id] = agent.x_star[agent_id]
        
        # Update position and x_dev
        p_est[agent_id] = p_hat[agent_id] + x_star[agent_id]

###     END Looping         - SCP Outer Loop



###     Plotting            - Static Position Estimates
print("\nPlotting")
print()

# Compare position estimates before and after reconstruction
fig1 = plt.figure(dpi=200)
ax1 = fig1.add_subplot(projection='3d')
ax1.set_title("Agent Position Estimates")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

for agent_id, agent in enumerate(agents): # Draw points
    ax1.scatter(p_est[agent_id][0], p_est[agent_id][1], p_est[agent_id][2], marker='*', c='m', label="After", s=100)
    ax1.scatter(p_hat[agent_id][0], p_hat[agent_id][1], p_hat[agent_id][2], facecolors='none', edgecolors='orangered', label="Before", s=100)
    ax1.scatter(p_true[agent_id][0], p_true[agent_id][1], p_true[agent_id][2], marker='x', c='g', label="True", s=100)
    #TODO: Fix labels on plot
    # ax1.text(p_true[agent_id][0], p_true[agent_id][1], p_true[agent_id][2], '%s' % (str(agent_id)))

for i, edge in enumerate(edges): # Draw edges
    p1 = p_est[edge[0]]
    p2 = p_est[edge[1]]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    ax1.plot(x, y, z, c='k', linewidth=1, alpha=0.5)[0]
plt.legend(["With Inter-agent Measurements", "Without Inter-agent Measurements", "True Position"], loc='best', fontsize=6, markerscale=0.4)
plt.grid(True)



###     Plotting            - Error Convergence
# Show convergence of estimated error vector to true error vector over time
#TODO: This needs to be fixed.
# x_norm_history = [x_norm_history[id].flatten() for i in range(num_agents)]
# plt.figure()
# for agent_id, agent in enumerate(agents):
#     label_str = "Agent " + str(agent_id)
#     plt.plot(total_iterations, x_norm_history[agent_id], label=label_str)
# plt.title("Convergence of Error Vector")
# plt.xlabel("Iterations")
# plt.ylabel("||x* - x||")
# # plt.ylim(left=0)
# plt.xlim((0, n_scp*n_admm))
# plt.legend(loc='best')
# plt.grid(True)



###     Plotting            - Animation
# Create position estimate over time data
p_hist = []
for id in range(num_agents):
    p_id = np.zeros((dim, n_iter))
    for iter in range(n_iter):
        p_id[:,iter] = p_hat[id].flatten() + x_history[id][:, iter]
    p_hist.append(p_id)

#TODO: Add animation
# fig, ax = plt.subplots()



###     Plotting            - Show Plots
plt.show()