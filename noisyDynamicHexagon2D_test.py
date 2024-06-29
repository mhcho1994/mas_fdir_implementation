"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - 2D Project
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
import matplotlib.animation as animation

from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from math import comb
from time import time


###     Imports             - User-Defined Files
from generic_agent import GenericAgent as Agent
from iam_models import distance



###     Initializations     - Scalars
dim             =   2   # 2 or 3
num_agents      =   6
num_faulty      =   1   # must be << num_agents for sparse error assumption
n_scp           =   20  # Number of SCP iterations
n_admm          =   6 # Number of ADMM iterations
n_iter          =   n_admm * n_scp
show_prob1      =   False
show_prob2      =   False
use_threshold   =   False
rho             =   0.5
iam_noise       =   0.00
warm_start      =   False
lam_lim         =   1
mu_lim          =   1
show_plots      =   False

###     Initializations     - Agents
# 6 agents making up a hexagon
agents      =   [None] * num_agents
d           =   10   # hexagon side length
agents[0]   =   Agent(agent_id= 0,
                      init_position= np.array([[d/2, -d*np.sqrt(3)/2]]).T)
agents[1]   =   Agent(agent_id= 1,
                      init_position= np.array([[-d/2, -d*np.sqrt(3)/2]]).T)
agents[2]   =   Agent(agent_id= 2,
                      init_position= np.array([[-d, 0]]).T)
agents[3]   =   Agent(agent_id= 3,
                      init_position= np.array([[-d/2, d*np.sqrt(3)/2]]).T)
agents[4]   =   Agent(agent_id= 4,
                      init_position= np.array([[d/2, d*np.sqrt(3)/2]]).T)
agents[5]   =   Agent(agent_id= 5,
                      init_position= np.array([[d, 0]]).T)

# Add error vector
agent_speed = 0.25
faulty_id = 5 #np.random.randint(0, high=num_agents)
fault_vec = np.array([[-0.2417], [0.2106]]) #agent_speed*2*(np.random.rand(dim, 1) - 0.5)
x_true = []
for id, agent in enumerate(agents):
    x_true.append(agent.error_vector)

# Circling
angular_vel = 2*np.pi / 100
circ_r = 2


# Set Neighbors
edges    = [[0,1], [0,2], [0,3], 
            [0,4], [0,5], [1,2],
            [1,3], [1,4], [1,5],
            [2,3], [2,4], [2,5],
            [3,4], [3,5], [4,5],
            
            [1,0], [2,0], [3,0], 
            [4,0], [5,0], [2,1],
            [3,1], [4,1], [5,1],
            [3,2], [4,2], [5,2],
            [4,3], [5,3], [5,4]] # these edges are directed

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
# Measurement model Phi
def meas_model(p, x_hat):
    measurements = []

    for edge in edges:
        dist = distance((p[edge[0]] + x_hat[edge[0]]), (p[edge[1]] + x_hat[edge[1]]))
        measurements.append(dist)

    return measurements


# True measurements
def true_meas(p):
    measurements = []

    for edge in edges:
        dist = distance((p[edge[0]]), (p[edge[1]])) + np.random.normal(scale=iam_noise)
        measurements.append(dist)

    return measurements

# Finds row of R
def get_Jacobian_row(edge_ind, p, x):
    edge = edges[edge_ind]
    agent1_id = edge[0]
    agent2_id = edge[1]
    pos1 = p[edge[1]] + x[edge[1]]
    pos2 = p[edge[0]] + x[edge[0]]
    disp    = (pos1 - pos2)
    R_k = np.zeros((1, dim*num_agents))

    dist = distance(pos1, pos2)
    R_k[:, dim*agent2_id:dim*(agent2_id + 1)] = disp.T  / dist
    R_k[:, dim*agent1_id:dim*(agent1_id + 1)] = -disp.T / dist

    return R_k

# Computes whole R matrix
def get_Jacobian_matrix(p, x):
    R = []

    for edge_ind, edge in enumerate(edges):
        R.append(get_Jacobian_row(edge_ind, p, x))
    
    return R

# Returns rank of Jacobian matrix
def get_Jacobian_rank(R):
    
    jac_matrix = R[0]
    for row in R[1:]:
        jac_matrix = np.vstack((jac_matrix, row))
    
    return np.linalg.matrix_rank(jac_matrix)

# Returns the matrix norm of the difference between old and new R matrices
def get_Jacobian_matrix_norm_diff(R_old, R_new):

    old_matrix = R_old[0]
    for row in R_old[1:]:
        old_matrix = np.vstack((old_matrix, row))

    new_matrix = R_new[0]
    for row in R_new[1:]:
        new_matrix = np.vstack((new_matrix, row))

    return np.linalg.norm(new_matrix - old_matrix)



###     Initializations     - Measurements and Positions

# Error
x_star = [np.zeros((dim, 1)) for i in range(num_agents)]                    # Equivalent to last element in x_history (below)
x_history = [np.zeros((dim, (n_iter))) for i in range(num_agents)]          # Value of x at each iteration of algorithm
x_norm_history = [np.zeros((1, (n_iter))) for i in range(num_agents)]       # Norm of difference between x_history and x_true

# Position
p_est = [agents[i].get_estimated_pos() for i in range(num_agents)]          # Will be updated as algorithm loops and err vector is reconstructed
p_hat = deepcopy(p_est)                                                     # Reported position of agent
p_orig = deepcopy(p_est)                                                    # CONSTANT: Original position of the agents
est_pos_history = [np.zeros((dim, n_iter)) for i in range(num_agents)]      # Value of p_hat at each iteration of algorithm
true_pos_history = [np.zeros((dim, (n_iter))) for i in range(num_agents)]   # Value of p at each iteration of algorithm

# Measurement
y = meas_model(p_hat, x_star)                                               # CONSTANT: Phi(p_hat + x_hat), true interagent measurement

# Residual
residuals = [np.zeros(n_iter) for i in range(num_agents)]                   # Running residuals of each agent (residual <= 1 is nominal)


###      Initializations    - Optimization Parameters
reset_lam = [False] * num_agents
reset_mu = [False] * num_agents
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
print("Faulty Agent Vector:", fault_vec.flatten())
print(f"Use Residual Threshold Optimization Skip: {use_threshold}")
print(f"Warm Start: {warm_start}")


###     Initializations     - Storing Dual Variables
lam_norm_history = [np.zeros((len(agents[i].get_edge_indices()), n_iter)) for i in range(num_agents)]
mu_norm_history = [np.zeros((len(agents[i].get_neighbors()), n_iter)) for i in range(num_agents)]
R_norm_diff_history = np.zeros(n_scp)
sum_err_rmse = 0
maximal_rank = dim*num_agents - comb(dim + 1, 2)

###     Looping             - SCP Outer Loop
print("\nLooping")
init_time = time()
for outer_i in tqdm(range(n_scp), desc="SCP Loop ", leave=False):

    exp_meas = meas_model(p_hat, x_star)

    R_new = get_Jacobian_matrix(p_hat, x_star)
    if (outer_i > 0):
        R_norm_diff = get_Jacobian_matrix_norm_diff(R, R_new)
        # print(f"\nR norm difference: {R_norm_diff}")
        R_norm_diff_history[outer_i] = R_norm_diff
    R = R_new

    jac_rank = get_Jacobian_rank(R)
    if (jac_rank != maximal_rank):
        print(f"\nMaximal Condition not met @ SCP Iteration {outer_i}")
        print(f"\tExpected: {maximal_rank}; Got: {jac_rank}")

    for agent in agents:
        agent.init_w(np.zeros((dim, 1)), agent.get_neighbors())


    ###     Looping             - ADMM Inner Loop
    for inner_i in tqdm(range(n_admm), desc="ADMM Loop", leave=False):

        for id, agent in enumerate(agents):
            angle = (inner_i + outer_i*n_admm)*angular_vel
            del_pos = np.array([[circ_r*np.cos(angle)], [circ_r*np.sin(angle)]])
            agent.position = p_orig[id] + del_pos
            p_hat[id] = agent.position
            est_pos_history[id][:, inner_i + outer_i*n_admm] = agent.position.flatten()
        
        x_true[faulty_id] = (inner_i + outer_i*n_admm)*fault_vec.flatten()


        p = deepcopy(p_hat)
        p[faulty_id] = deepcopy(p[faulty_id].flatten() + (inner_i + outer_i*n_admm)*fault_vec.flatten()).reshape(-1, 1) # True position of the agents
        y = true_meas(p)
        z = [(y[i] - meas) for i, meas in enumerate(exp_meas)]

        ##      Minimization        - Primal Variable 1
        for agent_id, agent in enumerate(agents):
            # Thresholding: Summation over edges
            term1 = 0
            for i, edge_ind in enumerate(agent.get_edge_indices()):
                R_k = R[edge_ind]
                constr_c = R_k[:, dim*agent_id:dim*(agent_id+1)] @ (-agent.x_star[agent_id]) - z[edge_ind]
                for nbr_id in agent.get_neighbors():
                    constr_c += R_k[:, dim*nbr_id:dim*(nbr_id+1)] @ agent.w[nbr_id]
                
                term1 += R_k[:, dim*agent_id:dim*(agent_id+1)].T @ (constr_c + (agent.lam[edge_ind] / rho))

            # Thresholding: Summation over neighbors
            term2 = 0
            for nbr_id in agent.get_neighbors():
                constr_d = -agent.x_star[agent_id] - agent.w[nbr_id]
                term2 += constr_d + (agent.mu[nbr_id] / rho)

            # Tresholding: Check threshold
            res = np.linalg.norm(term1 + term2)
            residuals[agent_id][inner_i + outer_i*n_admm] = res * rho
            if use_threshold and ((res * rho) <= 1):
                agent.x_bar = deepcopy(-agent.x_star[agent_id])
            else:
            # Optimization: Find x_bar if over threshold
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
            
            # Store: Reconstructed Error
            new_x = deepcopy(agent.x_bar.flatten()) + deepcopy(x_star[agent_id].flatten())
            x_history[agent_id][:, inner_i + outer_i*n_admm] = new_x.flatten()

            # Store: Convergence of Reconstructed Error Vector
            new_x_norm = np.linalg.norm(new_x.flatten() - x_true[agent_id].flatten())
            x_norm_history[agent_id][:, inner_i + outer_i*n_admm] = new_x_norm
            sum_err_rmse += new_x_norm


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

                if (not warm_start) and (np.linalg.norm(constr_c) > lam_lim):
                    # print(f"RESET LAM: Agent {agent_id} at SCP {outer_i}")
                    reset_lam[agent_id] = True
                new_lam = agent.lam[edge_ind] + rho * constr_c
                agent.lam[edge_ind] = deepcopy(new_lam)
                lam_norm_history[agent_id][i, (inner_i + outer_i*n_admm)] = np.linalg.norm(deepcopy(new_lam))

            # Summation for d() constraint
            for _, nbr_id in enumerate(agent.get_neighbors()):
                constr_d = agent.x_bar - agent.w[nbr_id]

                if (not warm_start) and (np.linalg.norm(constr_d) > mu_lim):
                    # print(f"RESET MU: Agent {agent_id} at SCP {outer_i}")
                    reset_mu[agent_id] = True
                new_mu = agent.mu[nbr_id] + rho * constr_d
                agent.mu[nbr_id] = deepcopy(new_mu)
                mu_norm_history[agent_id][i, (inner_i + outer_i*n_admm)] = np.linalg.norm(deepcopy(new_mu))


        ##      Store           - Position and Error Vectors
        for id, agent in enumerate(agents):
            # True Position
            true_pos_history[id][:, inner_i + outer_i*n_admm] = p[id].flatten()
    
                

    ###     END Looping         - ADMM Inner Loop
    
    # Update Error Vectors after ADMM subroutine
    for agent_id, agent in enumerate(agents): 
        for list_ind, nbr_id in enumerate(agent.get_neighbors()):
            agent.x_star[nbr_id] = agent.x_star[nbr_id] + agents[nbr_id].x_bar
        
        agent.x_star[agent_id] = agent.x_star[agent_id] + agent.x_bar
        x_star[agent_id] = agent.x_star[agent_id]
        
        # Update position and x_dev
        p_est[agent_id] = p_hat[agent_id] + x_star[agent_id]

    # Check if a reset flag was set
    for agent_id, agent in enumerate(agents):
        if (reset_lam[agent_id] or reset_mu[agent_id]):
            # print(f"RESET DUAL: Agent {agent_id} at SCP {outer_i}")
            agent.init_lam(np.zeros((1, 1)), agent.get_edge_indices())
            agent.init_mu(np.zeros((dim, 1)), agent.get_neighbors())
            reset_mu[agent_id] = False
            reset_lam[agent_id] = False

###     END Looping         - SCP Outer Loop
final_time = time()


###     Plotting            - Static Position Estimates
print("\nPlotting")
print()

# # Compare position estimates before and after reconstruction
# plt.figure()
# plt.title("Agent Position Estimates")
# plt.xlabel("x")
# plt.ylabel("y")
# for agent_id, agent in enumerate(agents):
#     plt.scatter(p_hat[agent_id][0], p_hat[agent_id][1], marker='o', c='c', label="Reported")
#     plt.scatter(p_est[agent_id][0], p_est[agent_id][1], marker='*', c='m', label="Reconstructed")
#     plt.scatter(true_pos_history[agent_id][0, -1], true_pos_history[agent_id][1, -1], marker='x', c='k', label="True")
# plt.legend(["Without Inter-agent Measurements", "With Inter-agent Measurements", "True Position"], loc='best', fontsize=5, markerscale=0.3)
# plt.grid(True)



###     Plotting            - Error Convergence
# Show convergence of estimated error vector to true error vector over time
x_norm_history = [x_norm_history[i].flatten() for i in range(num_agents)]
plt.figure(dpi=200)
for agent_id, agent in enumerate(agents):
    label_str = "Agent " + str(agent_id)
    plt.plot(total_iterations, x_norm_history[agent_id], label=label_str)
plt.title("Convergence of Error Vector")
plt.xlabel("Iterations")
plt.ylabel("||x* - x||")
plt.xlim((0, n_scp*n_admm))
plt.legend(loc='best')
plt.grid(True)

print(f"Num ADMM: {n_admm}")
print(f"Average RMSE: {sum_err_rmse / num_agents}")
print(f"Elapsed Time: {final_time - init_time}")



###     Plotting            - Animation
# Create position estimate over time data
p_hist = []
for id in range(num_agents):
    this_pos = np.zeros((dim, n_iter))
    for iter in range(n_iter):
        angle = iter*angular_vel
        del_pos = np.array([[circ_r*np.cos(angle)], [circ_r*np.sin(angle)]])

        this_pos[:,iter] = p_orig[id].flatten() + x_history[id][:, iter].flatten() + del_pos.flatten()
    p_hist.append(this_pos)

# Start figure
fig, ax = plt.subplots(dpi=200)
ax.set_title("Agent Estimated Position")
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.set_xlim((-20, 20))
ax.set_ylim((-20, 20))
scat_pos_est = [None] * num_agents # Position estimate during reconstruction
scat_pos_hat = [None] * num_agents # Initial position estimate
scat_pos_true = [None] * num_agents # True positions
line_pos_est = [None] * len(edges) # Inter-agent communication

# Draw each agent's original estimated, current estimated, and true positions
for agent_id, _ in enumerate(agents):
    scat_pos_est[agent_id] = ax.scatter(p_hist[agent_id][0, 0], p_hist[agent_id][1, 0], marker='*', c='c', label="After", s=100)
    scat_pos_hat[agent_id] = ax.scatter(est_pos_history[agent_id][0, 0], est_pos_history[agent_id][1, 0], facecolors='none', edgecolors='orangered', label="Before", s=100)
    scat_pos_true[agent_id] = ax.scatter(true_pos_history[agent_id][0, 0], true_pos_history[agent_id][1, 0], marker='x', c='g', label="True", s=100)

# Draw line for each edge of network
for i, edge in enumerate(edges):
    p1 = p_hist[edge[0]][:, 0]
    p2 = p_hist[edge[1]][:, 0]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    line_pos_est[i] = ax.plot(x, y, c='k', linewidth=1, alpha=0.5)[0]
ax.legend(["With Inter-agent Measurements", "Without Inter-agent Measurements", "True Position"], loc='best', fontsize=6, markerscale=0.4)
ax.grid(True)

# Update function
def update_pos_plot(frame):
    updated_ax = []
    # Draw each agent's original estimated, current estimated, and true positions
    for agent_id, _ in enumerate(agents):
        scat_pos_est[agent_id].set_offsets(p_hist[agent_id][:, frame])
        scat_pos_hat[agent_id].set_offsets(est_pos_history[agent_id][:, frame])
        scat_pos_true[agent_id].set_offsets(true_pos_history[agent_id][:, frame])
        updated_ax.append(scat_pos_hat[agent_id])
        updated_ax.append(scat_pos_est[agent_id])
        updated_ax.append(scat_pos_true[agent_id])
    
    # Draw line for each edge of network
    for i, edge in enumerate(edges):
        p1 = p_hist[edge[0]][:, frame]
        p2 = p_hist[edge[1]][:, frame]
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        line_pos_est[i].set_xdata(x)
        line_pos_est[i].set_ydata(y)
        updated_ax.append(line_pos_est[i])
    
    return updated_ax

# Call update function
pos_ani = animation.FuncAnimation(fig=fig, func=update_pos_plot, frames=n_iter, interval=100)
dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fname = "fig/2D-NoisyDynamicHexagon/pos2D_ani_" + dt_string + ".gif"
pos_ani.save(filename=fname, writer="pillow")



###     Plotting            - Residuals and Threshold

# # Start figure
# fig2, ax2 = plt.subplots(dpi=200)
# ax2.set_title("Residual monitor")
# ax2.set_xlabel("Iteration")
# ax2.set_ylabel("Residual")

# # Plot residuals of each agent
# for id, this_res_hist in enumerate(residuals):
#     ax2.plot(np.arange(n_iter), this_res_hist, label=f"Agent {id}")
# ax2.plot(range(n_iter), [1]*n_iter, label=f"Threshold")
# ax2.legend(loc='best')
# ax2.set_ylim(bottom=0, top=10)
# ax2.grid(True)


###     Plotting            - Dual Variables: Lambda

# fig_lam, ax_lam = plt.subplots(dpi=200)
# ax_lam.set_title(f"Lambda for agents; Faulty ID: {faulty_id}")
# ax_lam.set_xlabel("Iteration")
# ax_lam.set_ylabel("Lambda")
# for id, _ in enumerate(agents):
#     for i in range(lam_norm_history[id].shape[0]):
#         ax_lam.plot(np.arange(n_iter), lam_norm_history[id][i, :].flatten(), label=f"Agent {id}, Edge {i}")
# # ax_lam.legend(loc='best')
# ax_lam.grid(True)


###     Plotting            - Dual Variables: Mu

# fig_mu, ax_mu = plt.subplots(dpi=200)
# ax_mu.set_title(f"Mu for agents; Faulty ID: {faulty_id}")
# ax_mu.set_xlabel("Iteration")
# ax_mu.set_ylabel("Mu")
# for id, _ in enumerate(agents):
#     for i in range(mu_norm_history[id].shape[0]):
#         ax_mu.plot(np.arange(n_iter), mu_norm_history[id][i, :].flatten(), label=f"Agent {id}, Neighbor {i}")
# # ax_mu.legend(loc='best')
# ax_mu.grid(True)


###     Plotting            - Norm Difference in R

fig_R_diff, ax_R_diff = plt.subplots(dpi=200)
ax_R_diff.set_title("Norm Difference between old R and new R")
ax_R_diff.set_xlabel("Outer Loop Iteration")
ax_R_diff.set_ylabel("|| R_old - R_new ||")
ax_R_diff.plot(np.arange(n_scp), R_norm_diff_history)
ax_R_diff.grid(True)


###     Plotting            - Show Plots
if show_plots:
    plt.show()