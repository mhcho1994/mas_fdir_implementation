"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay, modified by MH Cho
Description:
            - Re-plot of the TAC paper simulation result
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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation

from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

###     Imports             - User-Defined Files
from generic_agent import GenericAgent as Agent
from iam_models import distance



###     Initializations     - Scalars
dim             =   3   # 2 or 3-dimensional
num_agents      =   20
num_faulty      =   4   # must be << num_agents for sparse error assumption
n_scp           =   6   # Number of SCP iterations
n_admm          =   10  # Number of ADMM iterations
n_iter          =   n_admm * n_scp
show_prob1      =   False
show_prob2      =   False



###     Initializations     - Agents
# 20 agents making up a complex 3d configuration
agents      =   [None] * num_agents
agents[0]   =   Agent(agent_id = 0, init_position = np.array([[0.1, 2.4, 5.4]]).T)
agents[1]   =   Agent(agent_id = 1, init_position = np.array([[2.8, 5.4, 6.1]]).T)
agents[2]   =   Agent(agent_id = 2, init_position = np.array([[2.15, 4.8, 4.3]]).T)
agents[3]   =   Agent(agent_id = 3, init_position = np.array([[1.15, 0.4, 3.9]]).T)
agents[4]   =   Agent(agent_id = 4, init_position = np.array([[3.0, 3.85, 5.4]]).T)
agents[5]   =   Agent(agent_id = 5, init_position = np.array([[3.4, 4.25, 2.0]]).T)
agents[6]   =   Agent(agent_id = 6, init_position = np.array([[3.45, 1.8, 2.2]]).T)
agents[7]   =   Agent(agent_id = 7, init_position = np.array([[5.2, 5.0, 5.25]]).T)
agents[8]   =   Agent(agent_id = 8, init_position = np.array([[5.3, 3.8, 0.1]]).T)
agents[9]   =   Agent(agent_id = 9, init_position = np.array([[5.2, 0.8, 3.15]]).T)
agents[10]  =   Agent(agent_id = 10, init_position = np.array([[6.2, 3.3, 5.6]]).T)
agents[11]  =   Agent(agent_id = 11, init_position = np.array([[5.05, 3.8, 3.6]]).T)
agents[12]  =   Agent(agent_id = 12, init_position = np.array([[4.15, 5.65, 3.4]]).T)
agents[13]  =   Agent(agent_id = 13, init_position = np.array([[0.15, 3.4, 2.45]]).T)
agents[14]  =   Agent(agent_id = 14, init_position = np.array([[1.85, 5.15, 0.65]]).T)
agents[15]  =   Agent(agent_id = 15, init_position = np.array([[2.4, 2.4, 1.6]]).T)
agents[16]  =   Agent(agent_id = 16, init_position = np.array([[1.4, 5.4, 2.4]]).T)
agents[17]  =   Agent(agent_id = 17, init_position = np.array([[3.2, 3.4, 0.2]]).T)
agents[18]  =   Agent(agent_id = 18, init_position = np.array([[5.4, 5.4, 1.4]]).T)
agents[19]  =   Agent(agent_id = 19, init_position = np.array([[4.7, 2.4, 5.4]]).T)



# Add error vector
faulty_id   =   np.array([0, 5, 7, 9, 10, 13],dtype=int)# may not work for some cases np.random.randint(low=0, high=num_agents, size=4)           
fault_vec   =   2*np.random.rand(dim,faulty_id.shape[0])-np.ones((dim,faulty_id.shape[0]))
for order, id in enumerate(faulty_id):
    agents[id].faulty = True
    agents[id].error_vector = fault_vec[:,order][:,np.newaxis]

x_true = []
for id, agent in enumerate(agents):
    x_true.append(agent.error_vector)


# Set Neighbors
edges       = [[0,2], [0,3], [0,4], [0,16], 
                [1,2], [1,4], [1,7], [1,11],
                [2,4], [2,5], [2,7], [3,4],
                [4,5], [4,6], [4,7], [4,10],
                [5,6], [5,8], [6,7], [6,9],
                [7,10], [8,9], [8,11], [9,11],
                [9,10], [10,11], [12,5], [12,7],
                [12,11], [12,2], [13,14], [13,15],
                [14,15], [3,15], [5,15], [13,0],
                [14,5], [6,14], [19,10], [19,4],
                [19,9], [18,8], [18,17], [18,11],
                [18,12], [17,14], [17,15], [17,8],
                [17,18], [16,14], [16,2], [16,13],
                [18,5], [15,6], [16,3]] 
edges_flip  =   deepcopy(edges)
for idx, dir_edge in enumerate(edges_flip):
    dir_edge.reverse()

edges       =   edges+edges_flip            # these edges are directed

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
print("Faulty Agent Vector: \n", fault_vec)


###     Looping             - SCP Outer Loop
print("\nStarting Loop")
for outer_i in tqdm(range(n_scp), desc="SCP Loop", leave=True):
    new_measurement = measurements(p_hat, x_star)
    z       =   [(y[i] - meas) for i, meas in enumerate(new_measurement)]
    R       =   get_Jacobian_matrix(p_hat, x_star)
    
    print('R temporary check:')
    print(len(R))
    print(R[0])

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

            x_history[agent_id][:, inner_i + outer_i*n_admm] = new_x.flatten()
            x_norm_history[agent_id][:, inner_i + outer_i*n_admm] = np.linalg.norm(new_x.flatten() - x_true[agent_id].flatten())

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
ax1.set_zlabel("z")

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
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

# Create position estimate over time data
p_hist = []
for id in range(num_agents):
    p_id = np.zeros((dim, n_iter))
    for iter in range(n_iter):
        p_id[:,iter] = p_hat[id].flatten() + x_history[id][:, iter]
    p_hist.append(p_id)

# Start figure
fig2 = plt.figure(dpi=300)
ax2 = fig2.add_subplot(projection='3d')
ax2.set_title("Agent Estimated Position")
ax2.set_xlabel("x position")
ax2.set_ylabel("y position")
ax2.set_zlabel("z position")
scat_pos_est = [None] * num_agents # Position estimate during reconstruction
scat_pos_hat = [None] * num_agents # Initial position estimakte
scat_pos_true = [None] * num_agents # True positions
line_pos_est = [None] * len(edges) # Inter-agent communication

# Draw each agent's original estimated, current estimated, and true positions
for agent_id, _ in enumerate(agents):
    scat_pos_est[agent_id] = ax2.plot(p_hist[agent_id][0, 0], p_hist[agent_id][1, 0], p_hist[agent_id][2, 0], 
                                        marker='*', c='c', linestyle='None', label="After", markersize=10)[0]
    scat_pos_hat[agent_id] = ax2.plot(p_hat[agent_id][0], p_hat[agent_id][1], p_hat[agent_id][2], 
                                        marker='o', markerfacecolor='none', c='orangered', linestyle='None', label="Before", markersize=10)[0]
    scat_pos_true[agent_id] = ax2.plot(p_true[agent_id][0], p_true[agent_id][1], p_true[agent_id][2], 
                                        marker='x', c='g', linestyle='None', label="True", markersize=10)[0]

# Draw line for each edge of network
for i, edge in enumerate(edges):
    p1 = p_hist[edge[0]][:, 0]
    p2 = p_hist[edge[1]][:, 0]
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    line_pos_est[i] = ax2.plot(x, y, z, c='gray', linewidth=1, alpha=0.25)[0]

arrow_prop_dict = dict(mutation_scale=4, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
for _, agents_id in enumerate(faulty_id):
    arrow_temp = Arrow3D([p_hat[agents_id][0,0], p_true[agents_id][0,0]], [p_hat[agents_id][1,0], p_true[agents_id][1,0]], [p_hat[agents_id][2,0], p_true[agents_id][2,0]], **arrow_prop_dict)
    ax2.add_artist(arrow_temp)
    
# ax2.legend(["With Inter-agent Measurements", "Without Inter-agent Measurements", "True Position"], loc='best', fontsize=6, markerscale=0.4)
ax2.grid(True)
ax2.set_aspect('equal')

# Update function
def update_pos_plot(frame):
    updated_ax = []
    # Draw each agent's original estimated, current estimated, and true positions
    for agent_id, _ in enumerate(agents):
        new_pos = (float(p_hist[agent_id][0, frame]), float(p_hist[agent_id][1, frame]), float(p_hist[agent_id][2, frame]))

        scat_pos_est[agent_id].set_data([new_pos[0]], [new_pos[1]])
        scat_pos_est[agent_id].set_3d_properties([new_pos[2]])

        updated_ax.append(scat_pos_est[agent_id])
    
    # Draw line for each edge of network
    for i, edge in enumerate(edges):
        p1 = p_hist[edge[0]][:, frame]
        p2 = p_hist[edge[1]][:, frame]
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        z = [p1[2], p2[2]]

        line_pos_est[i].set_data(x, y)
        line_pos_est[i].set_3d_properties(z)

        updated_ax.append(line_pos_est[i])
    
    return updated_ax
    
# Call update function
pos_ani = animation.FuncAnimation(fig=fig2, func=update_pos_plot, frames=n_iter, interval=100, blit=False, repeat=True)
dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fname = "fig/3D-Hexagon/pos3D_ani_" + dt_string + ".gif"
pos_ani.save(filename=fname, writer="pillow")



###     Plotting            - Show Plots
plt.show()