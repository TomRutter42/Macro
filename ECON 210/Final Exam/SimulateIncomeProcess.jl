# set current folder as the working directory
cd(@__DIR__)

# source in Primitives.jl 
include("Primitives.jl")

# Set seed 

Random.seed!(42)

# --------------------- #

## Specify the process for log(xt) - μ, then add μ back in and exponentiate.
## Use tauchen() from QuantEcon. 

### Specify the number of grid points for the discretization.

NTauchen_Sims = 101

## Assert than NTauchen_Sims is odd, throwing an error if not. 

odd_error_message = 
    "Number of grid points for discretization must be odd."

NTauchen_Sims % 2 == 1 ? nothing :
    throw(AssertionError(odd_error_message))

### Specify the number of standard deviations to include in the discretization.
### We want the lowest income state to be around 7000, as suggested by the Q. 
### Note sd of an AR(1) is σ_ε / sqrt(1 - ρ^2).
### log(7000) - log(50000) ≈ -2 
### and the standard deviation, given the parameters, is 
### σ_ε / sqrt(1 - ρ^2) = 0.12 / sqrt(1 - 0.95^2) ≈ 0.38
### 2 / 0.38 = 5.26

SDs = 5.1

### Specify the discretization.

log_x_minus_μ = tauchen(NTauchen_Sims, ρ, σ_ε, 0, SDs)

### Add μ back in, then exponentiate.

x_states = exp.(log_x_minus_μ.state_values .+ μ)
Π_x = log_x_minus_μ.p

# Stationary distribution - tested with higher powers. 
stat_dist = stationary_distributions(log_x_minus_μ)[1]

## Simulate a process of length T for x. 

### Specify the number of simulations to run.

NSimulations = 10

### Simulate the process for log x - log_x_minus_μ
log_x_minus_μ_T = zeros(NSimulations, T)

for i in 1:NSimulations
    # Pick a random initial state with probabilities according to stat_dist
    init_state_i = 
        sample(1:NTauchen_Sims, Weights(stat_dist))
        
    log_x_minus_μ_T[i, :] = 
        simulate(log_x_minus_μ, T, init = init_state_i)
end

## Convert each simulation to x.

xt_sims = exp.(log_x_minus_μ_T .+ μ)

## Convert each entry to Y_t = x_t * A_t.
Yt_sims = zeros(NSimulations, T)
for i in 1:NSimulations
    Yt_sims[i, :] = xt_sims[i, :] .* A
end

## Plot the simulated process for Y_t.
## Do not use standard form on the y-axis. 

Expected_X = transpose(stat_dist) * x_states

plot(t_vals, 
     transpose(Yt_sims), 
     xlabel = "Age", 
     ylabel = "Income",
     legend = :none, 
     xticks = collect(life_begins:10:(T - 1 + life_begins)), 
     formatter = :plain, 
     ylim = (0, maximum(Yt_sims)))

## Add the deterministic component of income to the plot.

plot!(t_vals, 
      A .* Expected_X, 
      label = "Deterministic Component of Income", 
      color = :black, 
      linestyle = :dash, 
      linewidth = 2)

## Save Graph

savefig("figures/Simulated_Income_Process.png")