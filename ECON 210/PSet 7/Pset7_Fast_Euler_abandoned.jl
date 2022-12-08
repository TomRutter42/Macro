# Solve Q1 of PSet 7. 

# set current folder as the working directory
cd(@__DIR__)

# ----------------------------------------------------------------------------- #

# Packages
using Distributed
using Interpolations
using Optim
using Plots
using QuantEcon

# ============================================================================= #

# Define Primitives:

## Utility function

function u(c, γ)
    # if c <= 0.0001
    #     return -100000
    # elseif γ == 1
    #     return log(c)
    # else
    #     return c^(1 - γ) / (1 - γ)
    # end
    return c^(1 - γ) / (1 - γ)
end

## Derivative of utility function

function u′(c, γ)
    # if c <= 0.0001
    #     return 1000
    # elseif γ == 1
    #     return 1/c
    # else
    #     return c^(-γ)
    # end
    return c^(-γ)
end

## Inverse of derivative of utility function

function u′_inv(MU, γ)
    return (MU)^(-1 / γ)
end

# ============================================================================= #

γ = 10 # risk aversion --- need to change
β = 0.9 
μ_x = 4.7 
σ_x = sqrt(0.01)
R_f = 1.01 
μ_r = 0.08 
σ_r = sqrt(0.11)

# ============================================================================= #

grid_size = 50

# Set up the fixed grid for wealth. 

w_grid = exp.(range(log(0.01), log(1000), length = grid_size)) 

# ============================================================================= #

# Discretize the income process. 

num_income_states = 7
income_chain = tauchen(num_income_states, 0, σ_x, μ_x, 3)
income_states = exp.(income_chain.state_values)
income_probs = income_chain.p[1, :]

# ============================================================================= #

# Discretize the returns process. 

num_return_states = 7
return_chain = tauchen(num_return_states, 0, σ_r, μ_r, 3)
return_states = exp.(return_chain.state_values)
return_probs = return_chain.p[1, :]

# ============================================================================= #
prob_matrix = zeros(num_income_states, num_return_states)
for i in 1:num_income_states
    for j in 1:num_return_states
        prob_matrix[i, j] = income_probs[i] * return_probs[j]
    end
end

# ============================================================================= #

# Define a function to find the wealth at which the borrowing constraint binds, 
# given V and the distribution of y. 

@everywhere function find_borrowing_constraint(V′_old_interp, u′_inv, γ, β, income_states, income_probs)
    
    # Calculate expected marginal value of wealth at zero wealth. 

    E_MU_W = β * sum(income_probs .* V′_old_interp.(income_states))

    w_bar = u′_inv(E_MU_W, γ) 
    
    return w_bar

end

# Define a function to calculate wealth distributions in the next period. 

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

# Define a function to find the new Value function given the old one. 

@everywhere function find_new_V(V_old_interp, V′_old_interp, u, u′_inv, γ, β, income_states, income_probs, return_states, return_probs, w_grid)
    
    # Initialize the consumption and portfolio savings policy functions. 

    V_new = zeros(grid_size) .- 10^50
    c_pol = zeros(grid_size)
    ϕ_pol = zeros(grid_size)

    # Step 1: find the critical wealth level at which the borrowing constraint binds.

    # display("Finding the borrowing constraint...")
    

    w_bar = find_borrowing_constraint(V′_old_interp, u′_inv, γ, β, income_states, income_probs)
    # println("w_bar = ", w_bar)

    # Loop over the wealth grid. 

    # Threads.@threads for (i, w) in enumerate(w_grid)

    for (i, wealth) in enumerate(w_grid) # can be parallelized

        # Step 2: For each w in the w_grid, if w is below the critical value at which 
        # the borrowing constraint binds, then the optimal consumption is equal to w and 
        # we arbitrarily set the optimal ϕ equal to one (there is no saving, so ϕ is undefined). 

        if wealth <= w_bar

            V_new[i] = u(wealth, γ) + β * mean(income_probs .* V_old_interp.(income_states))
            c_pol[i] = wealth
            ϕ_pol[i] = 1
            # println("w_bar = ", w_bar)
            # println("w = ", wealth, " is below the borrowing constraint. ")

        else 

            # println("w_bar = ", w_bar)
            # println("w = ", wealth, " is ABOVE the borrowing constraint. ")

            # Step 3: If w is above the critical value at which the borrowing constraint 
            # binds, then we are looking for an interior solution in consumption, but 
            # \phi must still be explicitly constrained to be between 0 and 1.   

            ## Step 3a: Define the objective function to be maximized. 

            function objective(x)
                cons = x[1]
                ϕ_share = x[2]
                w′_possible = wealth_states(ϕ_share, cons, wealth, R_f, income_states, income_probs, return_states, return_probs)

                E_V = sum(prob_matrix .* V_old_interp.(w′_possible))
                overall_util = u(cons, γ) + β * E_V

                return (-1.0) * overall_util
            end

            # opt = Opt(:LN_COBYLA, 2)

            # opt.max_objective = objective
            # opt.lower_bounds = [w_grid[1], 0.0]
            # opt.upper_bounds = [wealth, 1.0]
            # opt.xtol_abs = 10^(-30)

            # println("Guess Outcome 1: ", (-1.0) * objective([wealth / 2, 0.9]))
            # println("Guess Outcome 2: ", (-1.0) * objective([wealth / 4, 0.9]))
            # println("Guess Outcome 3: ", (-1.0) * objective([w_bar, 0.9]))

            # (maxf, maxpol, ret) = optimize(opt, [wealth / 2, 0.9])
            # println("got $maxf at $maxpol[1] and $maxpol[2] after $count iterations (returned $ret)")

            lower = [w_grid[1] / 2.0, 0.0]
            upper = [wealth, 1.0]
            initial_x = [w_bar, 0.91]
            
            results = optimize(objective, lower, upper, initial_x)

            V_new[i] = (-1.0) * Optim.minimum(results)
            c_pol[i] = Optim.minimizer(results)[1]
            ϕ_pol[i] = Optim.minimizer(results)[2]

            # println("Calculated Optimal Outcome: ", V_new[i])

            ## Step 3b: Define the constraints on c and \phi. 

            # lower = [0.0001, 0]
            # upper = [wealth, 1]

            # function()

            # ## Step 3c: Define the initial guess. 

            # initial_guess = [wealth / 2, 0.9]

            # println("Guess Outcome: ", objective(initial_guess[1], initial_guess[2]))
            # println("Guess Outcome: ", objective(initial_guess[1] / 2, initial_guess[2]))

            # ## Step 3d: Find the optimal c and \phi.
            # res = optimize(pol -> -objective(pol[1], pol[2]), lower, upper, initial_guess, Fminbox())

            # V_new[i] = -res.minimum 
            
            # c_pol[i] = res.minimizer[1]
            # ϕ_pol[i] = res.minimizer[2]
            # println("Optimum Outcome: ", objective(c_pol[i], ϕ_pol[i]))
            # println("Should be same optimum: ", V_new[i])

        end 

    end

    return V_new, c_pol, ϕ_pol

end

# ============================================================================= #

# Set up Main Loop 

# V_star = zeros(grid_size)
# c_star = zeros(grid_size)
# ϕ_star = zeros(grid_size)


## Step 2: While above some tolerance, find the new value function and update the old one.

function VFI(u, u′, u′_inv, γ, β, income_states, income_probs, return_states, return_probs, w_grid, tol)

    local V_star 
    local c_star
    local ϕ_star

    dist = 1 + tol
    j = 0

    grid_size = length(w_grid)

    ### Set up the initial guess for the value function.
    ### Use zeros

    V_old = u.(w_grid, γ)

    ### Form a PCHIP interpolant for V_old.

    V_old_interp = extrapolate(interpolate(w_grid, V_old, LinearMonotonicInterpolation()), Interpolations.Flat())

    ### Also specify the derivative of V_old_interp

    V′_old = u′.(w_grid, γ)
    V′_old[end] = 0

    V′_old_interp = extrapolate(interpolate(w_grid, V′_old, LinearMonotonicInterpolation()), Interpolations.Flat())

    while dist > tol
    
        V_new, c_opt, ϕ_opt = find_new_V(V_old_interp, V′_old_interp, u, u′_inv, γ, β, income_states, income_probs, return_states, return_probs, w_grid)
        V_star = deepcopy(V_new)
        c_star = deepcopy(c_opt)
        ϕ_star = deepcopy(ϕ_opt)
        # Calculate distance between V_new and V_old.
        j = j + 1
        println("Iteration: ", j)
        dist = deepcopy(maximum(abs.(V_new - V_old)))
        println("Distance: ", dist)
        # println(" ")
        
        # println("Old V")
        # display(V_old)

        # println("New V")
        # display(V_new)

        V_new_interp = extrapolate(interpolate(w_grid, V_new, LinearMonotonicInterpolation()), Interpolations.Flat())
        
        V_old = deepcopy(V_new)
        c_opt = deepcopy(c_opt)
        # println("c_opt")
        # display(c_opt)
        # println("phi_opt")
        # display(ϕ_opt)
        ϕ_opt = deepcopy(ϕ_opt)
        V_old_interp = deepcopy(V_new_interp)

        # Get derivative as well. 
        ## Calculate derivative by taking the slope to the right of each point.

        for i in 1:(grid_size - 1)
            V′_old[i] = (V_old[i + 1] - V_old[i]) / (w_grid[i + 1] - w_grid[i])
        end

        V′_old[end] = 0

        ## Form a monotonic interpolant for V′_old.

        V′_old_interp = extrapolate(interpolate(w_grid, V′_old, LinearMonotonicInterpolation()), Interpolations.Flat())

        # println("V'_old is")
        # display(V′_old)
        # println("V'_old_interp is")
        # display(V′_old_interp)

    end

    return V_star, c_star, ϕ_star 

end

tol = 10^(-35)
V_test, c_test, ϕ_test = VFI(u, u′, u′_inv, γ, β, income_states, income_probs, return_states, return_probs, w_grid, tol)
plot(w_grid[30:end], V_test[30:end], label = "Value Function")
# plot(w_grid[2:end], V_test[2:end], label = "Value Function")
plot(w_grid, c_test, label = "Consumption Policy Function")
# plot(w_grid, ϕ_opt, label = "Risk-Share Policy Function")
