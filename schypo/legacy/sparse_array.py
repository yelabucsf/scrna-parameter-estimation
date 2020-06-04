"""
    sparse_array.py
    Custom sparse array, since there is un-necessary overhead for maintaining 1d versions of the SciPy sparse matrices.
"""


import numpy as np


class sparse_array(object):
    """ Sparse array object. It cannot handle indexing like numpy arrays. """
    
    def __init__(self):
        
        self.values = {}
        
    def __getitem__(self, key):
        
        if np.issubdtype(type(key), np.integer): # key is a number
            return self.values[key] if key in self.values else np.nan
#         elif type(key) in [np.ndarray, list] and key.shape[0] == 1: # key is an np array containing 1 value
#             return self.values[key[0]] if key[0] in self.values else np.nan
        
        return np.array([self.values[k] if k in self.values else np.nan for k in key])
    
    def __setitem__(self, key, value):
        
        if np.issubdtype(type(key), np.integer): # any integer, meaning a single number
            self.values[key] = value
        else:
            
            if np.issubdtype(type(value), np.number): # any value, meaning a single number
                for k in key:
                    self.values[k] = float(value)
            else:
                for k,val in zip(key, value):
                    self.values[k] = val