## NOT TESTED

# Value Function Iteration 

using PyPlot

# == Parameters == #

# Discount factor

beta = 0.95
R = 1.05 

# == Grid == #

# Number of grid points

n = 100

# Minimum and maximum values of the asset a.

a_min = 0.1

a_max = 10.0

# Create a grid of asset values.

agrid = linspace(a_min, a_max, n)

# == Utility Function == #

# Utility function

function u(c)
    if c <= 0
        return -1e10
    else
        return log(c)
    end
end

### For every a_j in a, compute T_{ij} = u(a_i - a_j / R) - beta * V(a_j)

function T!(out, a, V)
    for (j, a_j) in enumerate(a)
        for (i, a_i) in enumerate(a)
            c = a_i - a_j / R
            out[i, j] = u(c) - beta * V[j]
        end
    end
end

# == Value Function Iteration == #

# Initial guess for the value function

V0 = zeros(n)

# Create an instance of the VFI type

vfi = VFI(T!, agrid, V0)

# Compute the value function

V = compute_fixed_point(vfi, verbose=true)

# == Plot == #

plot(agrid, V)

title("Value Function")

xlabel("a")

ylabel("V(a)")

show()

# == Compute the policy function == #

# Create a matrix to store the policy function

g = zeros(n)

# Compute the policy function

for (i, a_i) in enumerate(agrid)
    g[i] = R * a_i - agrid[argmax(V)]
end

# == Plot == #

plot(agrid, g)

title("Policy Function")

xlabel("a")

ylabel("g(a)")

show()


