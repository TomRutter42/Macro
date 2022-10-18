# Problem Set 2, ECON 210

# Tom Rutter 

# --------------------------------------------------------------------

# Import Packages 
import numpy as np 
import matplotlib.pyplot as plt
# from numba import jit 
# @jit(nopython=True)

# --------------------------------------------------------------------

# Define Parameters

T = 80 
beta = 0.98 
R = 1 / beta 
sigma = 0.5 

# --------------------------------------------------------------------

# Define the state-space grid 
# i.e., each entry corresponds to a possible value of wealth a

grid_max = 100.0
grid_size = 1000
## linspace returns evenly spaced numbers over a specified interval 
grid = np.linspace(0.01, grid_max, grid_size)

# --------------------------------------------------------------------

# Utility function 
def u(c): 
    return(c ** (1 - (1 / sigma)) / (1 - (1 / sigma)))

# --------------------------------------------------------------------

# For each period t, we will have a value function V_t (a). 
# We will store these value functions in a matrix V, where each row
# corresponds to a different period t.

V = np.zeros((T, grid_size))

# Similarly, store the policy rules in a matrix, 
# where each row corresponds to a different period t.

policy = np.zeros((T, grid_size))

# Initialize the value function at the final period T
# Note than in the final period, the agent just consumes all 
# of his wealth and gets utility u(c_T) = u(a_T). 
V[T - 1, :] = u(grid)
policy[T - 1, :] = grid

# Now we will iterate backwards from period T-1 to 0. 
# At each period t, we will compute the value function V_t (a). 
# We will store these value functions in a matrix V, where each row
# corresponds to a different period t.
# We will also store the policy function in a policy matrix, where
# each row corresponds to a different period t.
# At each period t, the value function is equal to the sum of the
# utility from the optimal policy c and the value function in the next period
# multiplied by beta. 

for t in range(T - 2, -1, -1):
    for i, a in enumerate(grid):

        # In state a, the agent can choose future state a_prime
        # which gives her utility u(a - R ** (-1) * a_prime) + beta * V_{t+1} (a_prime)
        # We will store these values in a vector called value
        # and then take the maximum value of this vector.

        value = np.zeros(grid_size)
        for j, a_prime in enumerate(grid):
            value[j] = u(a - R ** (-1) * a_prime) + beta * V[t + 1, j]

        # We must have that a_prime <= R * a, so we will set the value of
        # a_prime > a to be negative infinity.
        value[grid > R * a] = -np.inf

        # Now we take the maximum value of the vector value
        # and store it in the matrix V
        V[t, i] = np.max(value)

        # We also store the optimal policy in the policy matrix
        policy[t, i] = grid[np.argmax(value)]
