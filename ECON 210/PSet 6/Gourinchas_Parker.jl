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
retirement = 80

T = retirement - adulthood 

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

a_grid = exp.(range(0, log(300), length = n)) .- 1

a_grid

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