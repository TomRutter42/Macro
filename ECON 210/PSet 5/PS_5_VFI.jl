using Dierckx
using Distributions
using LinearAlgebra
using NLsolve
using Pkg
using Plots
using QuantEcon

# ----------------------------------------------------------------------------- #

# set current folder as the working directory
cd(@__DIR__)

# ============================================================================= #

# Step 1: Replicate Figure 1 of Deaton (1991)

# Figure 1 of Deaton (1991) is a plot of the consumption function for a 
# range of utility functions and income dispersions. 

# ----------------------------------------------------------------------------- #

# Define the CRRA utility function 

"""
    u(c, γ)

The CRRA utility function.
"""
function u(c, γ)
    if c < 0.01 
        return -100000
    elseif γ == 1
        return log(c)
    else
        return c^(1 - γ) / (1 - γ)
    end
end

# ----------------------------------------------------------------------------- #

# Define the grids we will use. 

### Define the grid sizes 

nx = 500 # number of grid points for cash-on-hand x
ns = 500 # number of grid points for saving s

## Define the grids. 
## Note that the saving grid implicitly contains the borrowing constraint. 
x_grid = range(50, 320, length = nx) # Grid for cash on hand
s_grid = range(0.0, 300, length = ns) # Grid for saving. 



# ----------------------------------------------------------------------------- #

# Define function to solve the infinite-horizon consumption savings problem with 
# a borrowing constraint. 

"""
    solve_csp(ρ, σ, μ, R, δ, b)

Solve the infinite-horizon consumption savings problem with a borrowing constraint
for a given value of relative risk aversion ρ, income dispersion σ, mean income μ,
interest rate R, discount rate δ, and borrowing constraint b.

Returns the consumption function, the value function, and the policy function. 
"""

function solve_csp(ρ, σ, μ, R, δ, x_grid, s_grid)

    # ρ is the coefficient of relative risk aversion
    # σ is the standard deviation of the income process
    # μ is the mean of the income process
    # R is the interest rate
    # δ is the discount rate

    # ----------------------------------------------------------------------------- #

    # Step (2): Form a discrete grid for the income process.
    ## Income is independent and identically distributed N(μ, σ^2). 

    ## Set n, which is the number of potential realizations of y 
    n = 100

    ## Set the bounds of the grid
    y_min = μ - 3 * σ
    y_max = μ + 3 * σ

    ## Form the grid

    y_grid = range(y_min, y_max, length = n)

    ## Form a probability grid 

    p_grid = [1/n for i in 1:n]

    ## Construct a grid of midpoints 

    y_midpoints = (y_grid[1:end-1] + y_grid[2:end]) / 2

    ## For each y in y_midpoints, compute the probability of observing discrete approximation y,
    ## which we take to be the value of the CDF at the midpoint minus the cdf at the prior midpoint, 
    ## filling in the boundaries. 

    p_grid[1] = cdf(Normal(μ, σ), y_midpoints[1])
    p_grid[2:end-1] = cdf(Normal(μ, σ), y_midpoints[2:end]) - cdf(Normal(μ, σ), y_midpoints[1:end-1])
    p_grid[end] = 1 - cdf(Normal(μ, σ), y_midpoints[end-1])

    # ----------------------------------------------------------------------------- #

    # Step (3): Guess a value function given wealth. We guess the value function is 0. 

    V_old = zeros(ns)

    ## Initialize the new value function and optimal savings function
    V_new = zeros(ns)
    s_opt = zeros(ns)
    c_opt = zeros(ns)

    # ----------------------------------------------------------------------------- #

    # Step (4): Update the value function, recursively, until within tolerance. 

    ## Set the tolerance level
    tol = 1e-5

    ## Set the maximum number of iterations
    max_iter = 10000

    ## Set the initial distance between the value functions to be greater than the tolerance
    dist = 10 * tol

    ## Set the iteration counter to 0
    iter = 0

    ## While tolerance is not met and the maximum number of iterations is not exceeded,
    ## update the value function.

    while dist > tol && iter < max_iter

        ## Update the iteration counter
        iter += 1
        println("Iteration: ", iter)

        ## For each possible value of cash-on-hand, calculate maximum attainable utility 
        ## and the corresponding optimal bond holdings.
        for (i, x) in enumerate(x_grid)

            ## Set up a vector with utility for each candidate saving choice 

            u_vec = zeros(ns)

            for (j, s) in enumerate(s_grid)

                ## Calculate the instantaneous utility of consumption x - s 

                u_inst = u(x - s, ρ)

                ## For each value of y, calculate the continutation utility from wealth Rs + y 

                u_cont = zeros(n)

                for (k, y) in enumerate(y_grid)

                    ## Find the index of the closest value of Rs + y in the grid
                    ind = findmin(abs.(x_grid .- (R * s + y)))[2]

                    ## Calculate the continuation utility
                    u_cont[k] = V_old[ind]

                end

                ## Calculate the expected continuation utility
                E_u_cont = sum(u_cont .* p_grid)

                ## Calculate the total utility, summing instantaneous
                ## and expected continuation utility. 

                u_total = u_inst + (1 / (1 + δ)) * E_u_cont

                ## Assign this total utility to the appropriate element of the utility vector
                u_vec[j] = u_total

            end

            ## Find the index of the maximum utility
            ind = findmax(u_vec)[2]

            ## Assign the maximum utility to the appropriate element of the value function
            V_new[i] = u_vec[ind]

            ## Assign the optimal bond holdings to the appropriate element of the policy function
            s_opt[i] = s_grid[ind]

            ## Assign the optimal consumption to the appropriate element of the consumption function
            c_opt[i] = x - s_opt[i]

        end

        ## Update the distance between the value functions
        dist = maximum(abs.(V_new - V_old))
        println("Difference in V's: ", dist)

        ## Update the old value function, which will be used in the next iteration

        V_old = copy(V_new)

    end

    ## Return the value function and the policy functions

    return V_new, s_opt, c_opt

end

# ----------------------------------------------------------------------------- #

# ## Solve the consumption savings problem for 
# ## ρ = 2, σ = 0.10, μ = 100, R = 1.05, δ = 0.10

# V, s_opt, c_opt = solve_csp(2, 0.10, 100, 1.05, 0.10, x_grid, s_grid)

# plot(x_grid, c_opt, label = "Consumption", xlabel = "Cash-on-hand", ylabel = "Consumption", legend = :topleft)

# ----------------------------------------------------------------------------- #

# Solve the Consumption savings case problem for the following four cases: 
## 1) ρ = 2, σ = 10 
## 2) ρ = 2, σ = 15 
## 3) ρ = 3, σ = 10
## 4) ρ = 3, σ = 15

## Case 1
V1, s_opt1, c_opt1 = solve_csp(2, 10, 100, 1.05, 0.10, x_grid, s_grid)

## Case 2
V2, s_opt2, c_opt2 = solve_csp(2, 15, 100, 1.05, 0.10, x_grid, s_grid)

## Case 3
V3, s_opt3, c_opt3 = solve_csp(3, 10, 100, 1.05, 0.10, x_grid, s_grid)

## Case 4
V4, s_opt4, c_opt4 = solve_csp(3, 15, 100, 1.05, 0.10, x_grid, s_grid)

# Plot the consumption policy functions all on the same graph. 

plot(x_grid, c_opt1, label = "ρ = 2, σ = 10", xlabel = "Cash-on-hand", ylabel = "Consumption", legend = :topleft)
plot!(x_grid, c_opt2, label = "ρ = 2, σ = 15")
plot!(x_grid, c_opt3, label = "ρ = 3, σ = 10")
plot!(x_grid, c_opt4, label = "ρ = 3, σ = 15")

# Save the plot: 
savefig("ConsumptionPolicyFunctions.png")

# ----------------------------------------------------------------------------- #

# Simulate an income process for 200 periods where μ = 100 
# and σ = 10. 

R = 1.05

## Draw 200 values from N(100, 10)

y_t = rand(Normal(100, 10), 200)

## Using the optimal saving function s_opt1, plot how wealth s + y 
## evolves over time.

s_t = zeros(200)
s_t[1] = s_opt1[findmin(abs.(x_grid .- (0 + y_t[1])))[2]]

for t in 2:200

    s_t[t] = s_opt1[findmin(abs.(x_grid .- (R * s_t[t-1] + y_t[t])))[2]]

end

## Using the optimal consumption function c_opt1, plot how consumption
## evolves over time.

c_t = zeros(200)

c_t[1] = c_opt1[findmin(abs.(x_grid .- (0 + y_t[1])))[2]]

for t in 2:200

    c_t[t] = c_opt1[findmin(abs.(x_grid .- (R * s_t[t-1] + y_t[t])))[2]]

end

## Plot the paths of income, saving, and consumption 

plot(y_t, label = "Income", xlabel = "Time", ylabel = "Income", legend = :topleft)
plot!(s_t, label = "Savings")
c_t_down = c_t .- 40
plot!(c_t_down, label = "Consumption - 40")

## Save the plot: 
savefig("IncomeSavingConsumption.png")