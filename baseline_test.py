"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
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
# from iam_models import distance



###     Initializations     - Scalars
dim             =   2   # 2 or 3
num_agents      =   6
num_faulty      =   1   # must be << num_agents for sparse error assumption
# num_edges       =   9   # equal to number of inter-agent measurements
n_scp           =   5  # Number of SCP iterations
n_admm          =   10  # Number of ADMM iterations

###     Initializations     - Agents
# 3 agents at vertices of equilateral triangle, 3 agents at midpoints of edges
agents                  =   [None] * num_agents
d                       =   6   # larger triangle side length
agents[0]               =   Agent(agent_id=     0,
                                  init_position=    np.array([[d/2,     0]]).T)
agents[1]               =   Agent(agent_id=     1,
                                  init_position=    np.array([[0,       0]]).T)
agents[2]               =   Agent(agent_id=     2,
                                  init_position=    np.array([[-d/2,    0]]).T)
agents[3]               =   Agent(agent_id=     3,
                                  init_position=    np.array([[-d/4,    d*np.sqrt(3)/4]]).T)
agents[4]               =   Agent(agent_id=     4,
                                  init_position=    np.array([[0,       d*np.sqrt(3)/2]]).T,
                                  faulty=           True,
                                  err_vector=       np.array([[1,   0]]).T)
agents[5]               =   Agent(agent_id=     5,
                                  init_position=    np.array([[d/4,     d*np.sqrt(3)/4]]).T)

x_true = []
for agent in agents:
    x_true.append(agent.error_vector)


# Set Neighbors
edges                   = [[0,1], [1,2], [2,3], 
                           [3,4], [4,5], [5,0],
                           [1,5], [1,3], [3,5],
                           [1,0], [2,1], [3,2], 
                           [4,3], [5,4], [0,5],
                           [5,1], [3,1], [5,3]] # these edges are directed
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
def measurements(p_hat, x_hat):
    measurements = []

    for edge in edges:
        disp    = ((p_hat[edge[1]] + x_hat[edge[1]]) - 
                   (p_hat[edge[0]] + x_hat[edge[1]]))
        measurements.append(0.5 * (disp.T @ disp))

    return measurements

# Finds row of R
def get_Jacobian_row(edge_ind):
    edge = edges[edge_ind]
    indices = [(edge[0]), (edge[1])]
    disp    = ((p_hat[edge[1]] + x_hat[edge[1]]) - 
                (p_hat[edge[0]] + x_hat[edge[0]]))
    R_k = np.zeros((1, dim*num_agents))

    R_k[:, dim*indices[1]:dim*(indices[1] + 1)] = disp.T
    R_k[:, dim*indices[0]:dim*(indices[0] + 1)] = -disp.T

    return R_k

# Computes whole R matrix
def get_Jacobian_matrix():
    R = []

    for edge_ind, edge in enumerate(edges):
        R.append(get_Jacobian_row(edge_ind))
    
    return R

###     Initializations     - Measurements and Positions
x_hat                   = [np.zeros((dim, 1))] * num_agents
p_true                  = [agents[i].get_true_pos() for i in range(num_agents)]
p_hat                   = [agents[0].get_estimated_pos(),
                           agents[1].get_estimated_pos(),
                           agents[2].get_estimated_pos(),
                           agents[3].get_estimated_pos(),
                           agents[4].get_estimated_pos(),
                           agents[5].get_estimated_pos()]       # Reported positions of agents
y                       = measurements(p_true, x_hat)           # Phi(p_hat + x_hat), true interagent measurement
p_est                   = deepcopy(p_hat)                       # Will be updated as algorithm loops and err vector is reconstructed



###      Initializations    - Optimization Parameters
rho = 1
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


###     Looping             - SCP Outer Loop
for outer_i in tqdm(range(n_scp), desc="SCP Loop", leave=False):
    new_measurement = measurements(p_hat, x_hat)
    z       =   [(y[i] - meas) for i, meas in enumerate(new_measurement)]
    R       =   get_Jacobian_matrix()

    for agent in agents:
        agent.init_w(np.zeros((dim, 1)), agent.get_neighbors())


    ###     Looping             - ADMM Inner Loop
    for inner_i in tqdm(range(n_admm), desc="ADMM Loop", leave=False):

        ##      Minimization        - Primal Variable 1
        for agent_id, agent in enumerate(agents):
            objective = cp.norm(agent.x_star[agent_id] + agent.x_cp)
            
            # Summation for c() constraint
            for list_ind, edge_ind in enumerate(agent.get_edge_indices()): 
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_cp - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w[agent_id]
                
                objective += ((rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[edge_ind].T @ (constr_c))
            
            # Summation for d() constraint
            for list_ind, nbr_id in enumerate(agent.get_neighbors()): 
                constr_d = agent.x_cp - agent.w[nbr_id]
                objective += ((rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob1 = cp.Problem(cp.Minimize(objective), [])
            prob1.solve()
            # assert prob1.status == cp.OPTIMAL, "Optimization problem not solved"
            agent.x_bar = np.array(agent.x_cp.value).reshape((-1, 1))

        ##      Minimization        - Thresholding Parameter
        # TODO: Implement

        ##      Minimization        - Primal Variable 2
        for agent_id, agent in enumerate(agents):
            objective = cp.norm(agent.x_star[agent_id] + agent.x_bar)

            for edge_ind in agent.get_edge_indices(): # summation for c() constraint
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c = constr_c + R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w_cp[agent_id]
                
                objective += ((rho/2)*cp.power(cp.norm(constr_c), 2)
                                + agent.lam[edge_ind].T @ (constr_c))
            
            for nbr_id in agent.get_neighbors(): # summation for d() constraint
                constr_d = agent.x_bar - agent.w_cp[nbr_id]
                objective += ((rho/2)*cp.power(cp.norm(constr_d), 2)
                              + agent.mu[nbr_id].T @ (constr_d))
                
            prob2 = cp.Problem(cp.Minimize(objective), [])
            prob2.solve()
            # assert prob2.status == cp.OPTIMAL, "Optimization problem not solved"
            for list_ind, nbr_id in enumerate(agent.get_neighbors()):
                agent.w[nbr_id] = np.array(agent.w_cp[nbr_id].value).reshape((-1, 1))


        ##      Multipliers         - Update Lagrangian Multipliers of Minimization Problem
        for agent_id, agent in enumerate(agents):
            
            for list_ind, edge_ind in enumerate(agent.get_edge_indices()): # summation for c() constraint
                constr_c = R[edge_ind][:, dim*agent_id:dim*(agent_id+1)] @ agent.x_bar - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R[edge_ind][:, dim*nbr_id:dim*(nbr_id+1)] @ agents[nbr_id].w[agent_id]
                
                agent.lam[edge_ind] = agent.lam[edge_ind] + rho * constr_c

            for list_ind, nbr_id in enumerate(agent.get_neighbors()): # summation for d() constraint
                constr_d = agent.x_bar - agent.w[nbr_id]
                agent.mu[nbr_id] = agent.mu[nbr_id] + rho * constr_d

    ###     END Looping         - ADMM Inner Loop
            
    for agent_id, agent in enumerate(agents): # Update Error Vectors after ADMM subroutine
        for list_ind, nbr_id in enumerate(agent.get_neighbors()):
            agent.x_star[nbr_id] = agent.x_star[nbr_id] + agents[nbr_id].x_bar
        agent.x_star[agent_id] = agent.x_star[agent_id] + agent.x_bar
        x_hat[agent_id] = agent.x_star[agent_id]
        p_est[agent_id] = p_hat[agent_id] + x_hat[agent_id] # Update position estimates using error vector

###     END Looping         - SCP Outer Loop


print("dun")
print()

### Plotting
plt.figure()
plt.title("Agent Position Estimates")
plt.xlabel("x")
plt.ylabel("y")
for agent_id, agent in enumerate(agents):
    plt.scatter(p_hat[agent_id][0], p_hat[agent_id][1], marker='o', c='c')
    plt.scatter(p_est[agent_id][0], p_est[agent_id][1], marker='*', c='m')
plt.grid(True)
plt.show()