"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Vishnu Vijay
Description:
            - Represents generic agent object
"""

import numpy as np

class GenericAgent():
    # Constructor
    def __init__(self, agent_id=0, init_position=np.zeros((3,1)), faulty=False, err_vector=np.zeros((3,1))):
        self.agent_id       =   agent_id
        self.position       =   init_position
        self.faulty         =   faulty 
        self.error_vector   =   err_vector
        self.neighbor_ids   =   None

    # Returns estimated position (true pos + error vector)
    def get_estimated_pos(self):
        if not self.faulty:
            return      self.position
        else:
            return      self.position + self.error_vector
    
    # Returns true position
    def get_true_pos(self):
        return self.position

    # Sets neighbors of current agent
    def set_neighbors(self, neighbor_list):
        self.neighbor_ids   =   neighbor_list
        return None

    # Returns list of agent ids that are neighbors
    def get_neighbors(self):
        return self.neighbor_ids
    