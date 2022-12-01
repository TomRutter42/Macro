# Problem Set 6; Question 1

# -------------------------------------------------------------

# Import packages

@everywhere using Distributed
@everywhere using Distributions
@everywhere using Interpolations
@everywhere using Plots
@everywhere using QuantEcon
@everywhere using Random
@everywhere using Statistics
@everywhere using StatsBase

# set current folder as the working directory
cd(@__DIR__)

# ------------------------------------------------------------

# Part 1: Approximate the income process with a Markov chain. 

## Specify an age profile for the income process.

adulthood = 20 
death = 80

T = death - adulthood 

A = zeros(T)

for t = 1:T
    A[t] = (-3.12 + 0.26 * (t + 20) - 0.0024 * (t + 20)^2 ) / 1.12
end

n = 3000
w_hat_grid = exp.(range(0, log(15), length = n)) .- 1

function u(c, γ)
    if c < 0.01 
        return -10^10 
    else
        return c^(1 - γ) / (1 - γ)
    end
end

β = 0.90 
γ = 4.0
R = 1.10

#### Define a value function matrix 

V = zeros(n, T)

#### Define a policy function matrix
C = zeros(n, T)

#### Now, induct backwards to solve the model.
#### In the final period before death T, agent dies at T+1, 
#### the agent consumes all of their wealth. 

V[:, T] = u.(w_hat_grid, γ) 
C[:, T] = w_hat_grid

## get epsilon draws and probs

num_draws = 7
σ = sqrt(0.02)
m = 3.0
mc_epsilon = tauchen(num_draws, 0, σ, 0.0, m)
eps = mc_epsilon.state_values
probs = mc_epsilon.p
### Take draws 
exp_draws = exp.(eps)
probs = probs[1, :]
max_w = maximum(w_hat_grid)

## function to get spread of wealth next period. 

# get_next_ws = function(w_hat, c_cand, exp_draws, t, A)
#     w_next = R .* (w_hat - c_cand) ./ exp_draws .+ A[t + 1]
#     return w_next
# end

## function to get optimal c given wealth 

@everywhere function get_optimal_c(w_hat, w_hat_grid, V_next_interp, exp_draws, probs, t, A, β, γ, R)

    # get the index of the last value of w_hat_grid that is less than w_hat
    ## we don't need to consider past this since cannot consume more than cash on hand
    max_index = searchsortedlast(w_hat_grid, w_hat)

    ## initialize the value function and policy function
    V_cands = zeros(max_index)
    c_cands = zeros(max_index)

    Threads.@threads for j in 1:max_index
            
        ### Take consumption in period T
        c_cand = w_hat_grid[j]
        
        ### Calculate income in next period 
        w_next = R .* (w_hat - c_cand) ./ exp_draws .+ A[t + 1]
        
        ### if any of the entries in w_next are greater than max(w_hat_grid),
        ### then set them to be max(w_hat_grid)

        w_next[w_next .> max_w] .= max_w

        ### Calculate candidate value function in period t
        V_cands[j] = u(c_cand, γ) + β * mean(V_next_interp.(w_next) .* exp_draws.^(1 - γ), weights(probs))

    end

    ## return the max of V_cands and the associated consumption 

    return maximum(V_cands), w_hat_grid[argmax(V_cands)]

end

## main function 

function backwards(t, V_next, w_hat_grid, exp_draws, probs, A, β, γ, R)

    V_t = zeros(n)
    C_t = zeros(n)

    max_w = maximum(w_hat_grid)

    ### Define a function to solve the problem in period t. 
    ### Inputs: 
    ### - t: the period in which we are solving the problem
    ### - V_next: the value function in the next period
    ### - w_hat_grid: the grid for normalized cash on hand

    ### Outputs: 
    ### - V: the value function in period t
    ### - c: the policy function in period t 

    ## Take a linear interpolation over V_next. 
    V_next_interp = LinearInterpolation(w_hat_grid, V_next)

    # pmap(get_optimal_c, w_hat_grid, Ref(w_hat_grid), Ref(V_next_interp), Ref(exp_draws), Ref(probs), Ref(t), Ref(A), Ref(β), Ref(γ), Ref(R))

    Threads.@threads for i in 1:n 

        ### Get the optimal consumption given w_hat_grid[i]
        V_t[i], C_t[i] = get_optimal_c(w_hat_grid[i], w_hat_grid, V_next_interp, exp_draws, probs, t, A, β, γ, R)

    end

    return V_t, C_t

end

for t in T-1:-1:1
    println(t)
    V[:, t], C[:, t] = backwards(t, V[:, t + 1], w_hat_grid, exp_draws, probs, A, β, γ, R)
end

### Save V and C to a file. 

using JLD
save("V.jld", "V", V)
save("C.jld", "C", C)
save("w_hat_grid.jld", "w_hat_grid", w_hat_grid)