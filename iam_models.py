import numpy as np

def distance(pos1, pos2):
    return np.linalg.norm(pos2 - pos1)

def displacement(pos1, pos2):
    return (pos2 - pos1)