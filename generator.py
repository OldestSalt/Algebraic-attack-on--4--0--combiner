import math
import numpy as np
import scipy.linalg as linalg
import sympy as sp
from itertools import combinations

class LFSR:
    def __init__(self, start_state, coefs_array):
        self.state = start_state
        self.feedback_matrix = np.eye(len(start_state) - 1, dtype = int)
        self.feedback_matrix = np.vstack((np.zeros(len(start_state) - 1, dtype = int), self.feedback_matrix))
        self.feedback_matrix = np.hstack((self.feedback_matrix, coefs_array.reshape(len(coefs_array), 1)))
        if (len(start_state) != len(coefs_array)):
            raise Exception
    def next(self):
        self.state = np.dot(self.state, self.feedback_matrix) % 2

class LMcombiner:
    def __init__(self, LFSRs, projection_matrix, output_function):
        self.feedback_matrix = linalg.block_diag(*[lfsr.feedback_matrix for lfsr in LFSRs])
        self.state = np.concatenate(tuple([lfsr.state for lfsr in LFSRs]))
        self.projection_matrix = projection_matrix
        self.output_function = output_function

    def calculate_output(self):
        return self.output_function(*np.dot(self.state, self.projection_matrix)) % 2

    def next(self):
        self.state = np.dot(self.state, self.feedback_matrix) % 2
        return self.calculate_output()

class SymLFSR:
    def __init__(self, start_state, coefs_array):
        self.state = start_state
        self.feedback_matrix = np.eye(len(start_state) - 1, dtype = int)
        self.feedback_matrix = np.vstack((np.zeros(len(start_state) - 1, dtype = int), self.feedback_matrix))
        self.feedback_matrix = np.hstack((self.feedback_matrix, coefs_array.reshape(len(coefs_array), 1)))
        if (len(start_state) != len(coefs_array)):
            raise Exception
    def next(self):
        self.state = np.array([sp.trunc(poly, 2) for poly in np.dot(self.state, self.feedback_matrix)])

class SymLMcombiner:
    def __init__(self, LFSRs, projection_matrix, output_function):
        self.feedback_matrix = linalg.block_diag(*[lfsr.feedback_matrix for lfsr in LFSRs])
        self.state = np.concatenate(tuple([lfsr.state for lfsr in LFSRs]))
        self.projection_matrix = projection_matrix
        self.output_function = output_function
        self.monomials = {math.prod(comb): ''.join([sym.name for sym in comb]) for comb in combinations(self.state, 2)}

    def calculate_output(self):
        output = sp.trunc(self.output_function(*np.dot(self.state, self.projection_matrix)), 2)
        output = output.subs(self.monomials)
        return output

    def next(self):
        self.state = np.array([sp.trunc(poly, 2) for poly in np.dot(self.state, self.feedback_matrix)])
        return self.calculate_output()