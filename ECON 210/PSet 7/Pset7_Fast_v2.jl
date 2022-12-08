# Solve Q1 of PSet 7. 

# set current folder as the working directory
cd(@__DIR__)

# ----------------------------------------------------------------------------- #

# Packages
using Distributed
@everywhere using Interpolations
# @everywhere using Optim
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

N = 200 # grid sized for cash on hand

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

        # Interpolate V_old

        @everywhere V_old_interp = LinearInterpolation(w_grid, V_old, extrapolation_bc = Flat())
        
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


@everywhere γ = 10
tol_10 = u(w_grid[end], γ) - u(w_grid[end - 1], γ)
V_10 = VFI(tol_10)
@everywhere γ = 5
tol_5 = u(w_grid[end], γ) - u(w_grid[end - 1], γ)
V_5 = VFI(tol_5)
@everywhere γ = 2
tol_2 = u(w_grid[end], γ) - u(w_grid[end - 1], γ)
V_2 = VFI(tol_2)

# Plot V 

plot(w_grid, V, label = "Value Function", lw = 2, legend = :topleft, xlabel = "Cash on Hand")

# Given V, solve for consumption and share of risky assets.

function solve_policy(V)

    # Initialize policy functions

    c_policy = zeros(length(w_grid))
    ϕ_policy = zeros(length(w_grid))
    constrained = zeros(length(w_grid))

    # Form a spline approximation to V, that is flat outside the grid
    # i.e., doesn't keep growing as w gets really large. 

    @everywhere V_interp = extrapolate(interpolate(w_grid, V, LinearMonotonicInterpolation()), Interpolations.Flat())

    Threads.@threads for (i, w) in collect(enumerate(w_grid))
                
        c_grid = range(w_grid[1], w, length = length(w_grid))
        
        V_candidates = zeros(length(c_grid), length(grid_ϕ))

        for (j, c) in enumerate(c_grid)

            for (k, ϕ) in enumerate(grid_ϕ)

                w′ = wealth_states(ϕ, c, w, R_f, income_states, income_probs, return_states, return_probs)

                V_candidates[j, k] = u(c, γ) + β * sum(prob_matrix .* V_old_interp.(w′))

            end

        end

        V_max = maximum(V_candidates)

        indices = findall(V_candidates .== V_max)

        c_policy[i] = c_grid[indices[1][1]]
        ϕ_policy[i] = grid_ϕ[indices[1][2]]

        if c_policy[i] == w
            constrained[i] = 1
        else 
            constrained[i] = 0
        end

    end

    return c_policy, ϕ_policy, constrained

end

@everywhere γ = 10
c_policy_10, ϕ_policy_10, constrained_10 = solve_policy(V_10)
@everywhere γ = 5
c_policy_5, ϕ_policy_5, constrained_5 = solve_policy(V_5)
@everywhere γ = 2
c_policy_2, ϕ_policy_2, constrained_2 = solve_policy(V_2)

# Plot policy functions

plot(w_grid, c_policy_10, label = "γ = 10", lw = 2, legend = :topleft, xlabel = "Cash on Hand", ylabel = "Consumption")
plot!(w_grid, c_policy_5, label = "γ = 5", lw = 2)
plot!(w_grid, c_policy_2, label = "γ = 2", lw = 2)
## Save 
savefig("c_policy.png")


# Plot phi policy function only for constrained == 0
indices_10 = findall(constrained_10 .== 0)
indices_5 = findall(constrained_5 .== 0)
indices_2 = findall(constrained_2 .== 0)
plot(w_grid[indices_10], ϕ_policy_10[indices_10], label = "γ = 10", lw = 2, legend = :topleft, xlabel = "Cash on Hand", ylabel = "ϕ")
plot!(w_grid[indices_5], ϕ_policy_5[indices_5], label = "γ = 5", lw = 2)
plot!(w_grid[indices_2], ϕ_policy_2[indices_2], label = "γ = 2", lw = 2)
savefig("phi_policy.png")




