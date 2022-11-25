# Problem Set 6; Question 1

# -------------------------------------------------------------

# Import packages

using QuantEcon
using Plots
using Random

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

### Normalize the first entry of the income process to be 20,000.

A = A / A[1] * 20000

## Plot the age profile A against an x-axis which is 
## the index of A plus 20. 
## No legend is needed.
plot(20:T+20-1, A, xlabel = "Age", title = "Income Profile", legend = false)


## Specify the transition matrix for the Markov chain.
## Use the Tauchen method to approximate the income process. 
## - go to 5 standard deviations from the mean
## We specify the process for ε_t

σ = 0.02
N = 40
m = 3

mc_epsilon = tauchen(N, 0, σ, 0.0, m)

## Run 10 simulations of the epsilon process, running the simulation 
## for T periods. 

### Define an array simulations: 
### - T columns 
### - 10 rows
### - each row is a simulation of the epsilon process

### set seed 
Random.seed!(7)

simulations = zeros(10, T)

for i = 1:10
    simulations[i, :] = simulate(mc_epsilon, T)
    ## No uncertainty in the first period. 
    simulations[i, 1] = 0
end

### Take the exponent of simulations 

simulations_exp = exp.(simulations)

### For each entry in simulations_exp, multiply by the entry to the left 
### to get a running product. 

sims_exp_running = simulations_exp

for i = 1:10
    for t = 2:T
        sims_exp_running[i, t] = sims_exp_running[i, t] * sims_exp_running[i, t-1]
    end
end

### Multiply each row of sims_exp_running by the corresponding entry in A.

sims_exp_running_A = sims_exp_running

for i = 1:10
    sims_exp_running_A[i, :] = sims_exp_running_A[i, :] .* A
end

### Plot each of the ten simulations of the income process on the same graph. 
### Set up the initial plot.
### No standard form on the y-axis, with commas for thousands.

plot(20:T+20-1, sims_exp_running_A[1, :], xlabel = "Age", 
     legend = false, yformatter = :plain, yaxis = ("Income", (0, 5.0 * 20000)))

for i = 2:10
    plot!(20:T+20-1, sims_exp_running_A[i, :])
end

## Hide the legend 

plot!(legend = false)

savefig("income_process_sims.png")

# ------------------------------------------------------------

# Solve finite-horizon Gourinchas-Parker model. 

### Step 1: Choose a grid for normalized cash on hand. 

n = 10
w_hat_grid = exp.(range(0, log(300)), length = n) .- 1
w_hat_grid

### Step 2: Specify Primitives

#### Utility is CRRA.

function u(c, γ)
    if c <= 0.001
        return -10^10
    else
        return (c^(1 - γ) - 1) / (1 - γ)
    end
end

#### Parameters: 

β = 0.90 
γ = 4.0
R = 1.10

#### Define a value function matrix 

V = zeros(n, T+1)

#### Define a policy function matrix
C = zeros(n, T)

#### In the final period, the value function is just utility of cash on hand.

V[:, T] = zeros(n)

#### Now, induct backwards to solve the model.
#### In the final period before death T, agent dies at T+1, 
#### the agent consumes all of their wealth. 

V[:, T] = u.(w_hat_grid, γ) 
C[:, T] = w_hat_grid

#### Now, induct backwards to solve the model.

##### Define a function to solve the problem in period t.

function backwards(t, V_next) 

    V_t = zeros(n)
    C_t = zeros(n)

    ### Define a function to solve the problem in period t. 
    ### Inputs: 
    ### - t: the period in which we are solving the problem
    ### - V_next: the value function in the next period
    ### - w_hat_grid: the grid for normalized cash on hand

    ### Outputs: 
    ### - V: the value function in period t
    ### - c: the policy function in period t 

    for i in 1:n 
        for j in 1:n 
            ### Define the cash on hand in period t
            w_hat = w_hat_grid[i]
            ### Take consumption in period T
            c_cand = w_hat_grid[j]
            ### Define the continuation utility 
            
            ### Define the value function in period t
            V_cand = u(c, γ) + β * V_next[j]
            ### If c_cand is greater than w_hat, then the borrowing constraint 
            ### is violated. 
            if c_cand > w_hat
                V_cand = -10^10
            end
            ### If the value function is higher than the current value function, 
            ### update the value function and the policy function. 
            if V_cand > V_t[i]
                V_t[i] = V_cand
                C_t[i] = c_cand
            end
        end




