# Solve Q1 of PSet 7. 

# set current folder as the working directory
cd(@__DIR__)

# ----------------------------------------------------------------------------- #

# Packages
@everywhere using Distributed
@everywhere using Interpolations
@everywhere using Optim
@everywhere using Plots
@everywhere using QuantEcon

# ============================================================================= #

# Define Primitives:

## Utility function

@everywhere function u(c, γ)
    return c^(1 - γ) / (1 - γ)
end

# ============================================================================= #

# γ = 10 # risk aversion --- need to change
β = 0.9 
μ_x = 4.7 
σ_x = sqrt(0.01)
R_f = 1.01 
μ_r = 0.08 
σ_r = sqrt(0.11)

# ============================================================================= #

# Discretize the income process. 

num_income_states = 2
income_chain = tauchen(num_income_states, 0, σ_x, μ_x, 3)
@everywhere income_states = exp.(income_chain.state_values)
@everywhere income_probs = income_chain.p[1, :]

# ============================================================================= #

# Discretize the returns process. 

num_return_states = 2
return_chain = tauchen(num_return_states, 0, σ_r, μ_r, 3)
@everywhere return_states = exp.(return_chain.state_values)
@everywhere return_probs = return_chain.p[1, :]

# ============================================================================= #

# Matrix of state-by-state probabilities. 
prob_matrix_init = zeros(num_income_states, num_return_states)
for i in 1:num_income_states
    for j in 1:num_return_states
        prob_matrix_init[i, j] = income_probs[i] * return_probs[j]
    end
end
@everywhere prob_matrix = deepcopy(prob_matrix_init)

# ============================================================================= #

# Define a function to find the wealth at which the borrowing constraint binds, 
# given V and the distribution of y. 

@everywhere function find_borrowing_constraint(V′_old_interp, u′_inv, γ, β, income_states, income_probs)
    
    # Calculate expected marginal value of wealth at zero wealth. 

    E_MU_W = β * sum(income_probs .* V′_old_interp.(income_states))

    w_bar = u′_inv(E_MU_W, γ) 
    
    return w_bar

end

# ============================================================================= #

# Grids 

N = 100 #number of elements on the grid for cash on hand, total savings
# N_ϕ = 100 #Number of elements on the grid for share risky

# Wealth, consumption grid spaced so more values closer to boundary

@everywhere w_grid = exp.(range(log(80), log(250), length = N)) 
@everywhere w_grid = append!([1.0, 20.0, 40.0, 60.0], w_grid)
@everywhere grid_ϕ = (range(0.0, 1.0, length = N)) .^ 0.5

# ============================================================================= #

@everywhere function wealth_states(ϕ, c, w, R_f, income_states, income_probs, return_states, return_probs)

    # Loop over the possible income states and return_states, and calculate the
    # corresponding wealth state. 

    w′ = zeros(num_income_states, num_return_states)

    for i in 1:num_income_states
        for j in 1:num_return_states
            w′[i, j] = (w - c) * (R_f + ϕ * (return_states[j] - R_f)) + income_states[i] 
        end
    end 
    
    return w′

end

# ============================================================================= #

# Define VFI 

function VFI(tolerance)

    counter = 0

    dist = tolerance + 1

    # Initialize value function

    @everywhere V = u.(w_grid, γ) 

    while dist > tolerance

        @everywhere V_old = deepcopy(V)
        
        Threads.@threads for (i, w) in collect(enumerate(w_grid))
                
            c_grid = range(w_grid[1], w, length = length(w_grid))
            
            V_candidates = zeros(length(c_grid), length(grid_ϕ))

            for (j, c) in enumerate(c_grid)

                for (k, ϕ) in enumerate(grid_ϕ)

                    w′ = wealth_states(ϕ, c, w, R_f, income_states, income_probs, return_states, return_probs)

                    V_candidates[j, k] = u(c, γ) + β * sum(prob_matrix .* V_old_interp.(w′))

                end

            end

        V[i] = maximum(V_candidates)

        end

        counter = counter + 1
        dist = maximum(abs.(V - V_old))

        println("Iteration: ", counter)
        println("Distance: ", dist)

    end 

    return V

end


@everywhere γ = 2.0
V_10 = VFI(10^(-30))

# Plot V 

plot(w_grid, V, label = "Value Function", lw = 2, legend = :topleft, xlabel = "Cash on Hand")

# Given V, solve for consumption and share of risky assets.

# function solve_policy(V)

#     # Initialize policy functions

#     c_policy = zeros(length(w_grid))
#     ϕ_policy = zeros(length(w_grid))

#     # Form a spline approximation to V, that is flat outside the grid
#     # i.e., doesn't keep growing as w gets really large. 

#     @everywhere V_interp = extrapolate(interpolate(w_grid, V, LinearMonotonicInterpolation()), Interpolations.Flat())

#     Threads.@threads for (i, w) in collect(enumerate(w_grid))

#         valid_indices = findall(w_grid .<= w) # indices of w_grid that are valid

#         c_grid = w_grid[valid_indices] # otherwise liquidity constraint is binding. 

#         ϕ_grid = grid_ϕ[valid_indices] # otherwise liquidity constraint is binding.
        
#         V_candidates = zeros(length(c_grid), length(ϕ_grid))

#         for (j, c) in enumerate(c_grid)

#             for (k, ϕ) in enumerate(ϕ_grid)

#                 w′ = wealth_states(ϕ, c, w, R_f, income_states, income_probs, return_states, return_probs)

#                 V_candidates[j, k] = u(c, γ) + β * sum(prob_matrix .* V_interp.(w′))

#             end

#         end

#         V_max = maximum(V_candidates)

#         indices = findall(V_candidates .== V_max)

#         c_policy[i] = c_grid[indices[1][1]]
#         ϕ_policy[i] = ϕ_grid[indices[1][2]]

#     end

#     return c_policy, ϕ_policy

# end

# c_policy, ϕ_policy = solve_policy(V_10)

# # Plot policy functions

# plot(w_grid, c_policy, label = "Consumption", lw = 2, legend = :topleft, xlabel = "Cash on Hand")

# plot(w_grid, ϕ_policy, label = "Share of Risky Assets", lw = 2, legend = :topleft, xlabel = "Cash on Hand")





