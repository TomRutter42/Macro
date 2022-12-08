# Solve Q1 of PSet 7. 

# set current folder as the working directory
cd(@__DIR__)

# ----------------------------------------------------------------------------- #

# Packages
using Distributed
using Optim
using Interpolations
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

γ = 0.5 # risk aversion
β = 0.9 
μ_x = 4.7 
σ_x = sqrt(0.01)
R_f = 1.01 
μ_r = 0.08 
σ_r = sqrt(0.11)

# ============================================================================= #

grid_size = 1000

# Set up the fixed grid for wealth. 

w_grid = exp.(range(0.01, log(400), length = grid_size)) .- 1

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

# Define a function to find the wealth at which the borrowing constraint binds, 
# given V and the distribution of y. 

@everywhere function find_borrowing_constraint(V′, u′_inv, γ, β, income_states, income_probs)
    
    # Calculate expected marginal value of wealth at zero wealth. 

    E_MU_W = β * sum(income_probs .* V′.(income_states))

    w_bar = u′_inv(E_MU_W, γ) 
    
    return w_bar

end

# Define a function to calculate wealth distributions in the next period. 

@everywhere function wealth_states(ϕ, w, c, R_f, income_states, income_probs, return_states, return_probs)

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

@everywhere function find_new_V(V_old, V′_old, u, u′_inv, γ, β, income_states, income_probs, return_states, return_probs, w_grid)
    
    # Initialize the consumption and portfolio savings policy functions. 

    V_new = zeros(grid_size)

    # Step 1: find the critical wealth level at which the borrowing constraint binds.

    w_bar = find_borrowing_constraint(V′_old, u′_inv, γ, β, income_states, income_probs)

    # Loop over the wealth grid. 

    # Threads.@threads for (i, w) in enumerate(w_grid)

    for (i, w) in enumerate(w_grid) # can be parallelized

        # Step 2: For each w in the w_grid, if w is below the critical value at which 
        # the borrowing constraint binds, then the optimal consumption is equal to w and 
        # we arbitrarily set the optimal ϕ equal to one (there is no saving, so ϕ is undefined). 

        if w <= w_bar

            V_new[i] = u(w, γ) + β * mean(income_probs .* V_old.(income_states))

        else 

            # Step 3: If w is above the critical value at which the borrowing constraint 
            # binds, then we are looking for an interior solution in consumption, but 
            # \phi must still be explicitly constrained to be between 0 and 1.   

            ## Step 3a: Define the objective function to be maximized. 

            function objective(c, ϕ)
                w′ = wealth_states(ϕ, w, c, R_f, income_states, income_probs, return_states, return_probs)
                E_V = transpose(income_probs) * V′_old.(w′) * return_probs
                U = u(c, γ) + β * E_V
                return -U
            end

            ## Step 3b: Define the constraints on c and \phi. 

            lower = [0.0001, 0]
            upper = [w, 1]

            ## Step 3c: Define the initial guess. 

            initial_guess = [w / 2, 0.9]

            ## Step 3d: Find the optimal c and \phi.
            res = optimize(p -> -objective(p[1], p[2]), lower, upper, initial_guess, Fminbox(NelderMead()))

            V_new[i] = -res.minimum

        end 

    end

    return V_new

end

# ============================================================================= #

# Set up Main Loop 

## Step 1: Set up the initial guess for the value function.

V_old = zeros(grid_size)

### Form a PCHIP interpolant for V_old.

V_old_interp = extrapolate(interpolate(w_grid, V_old, SteffenMonotonicInterpolation()), Interpolations.Flat())

### Also specify the derivative of V_old_interp

V′_old = zeros(grid_size)

V′_old_interp = extrapolate(interpolate(w_grid, V′_old, SteffenMonotonicInterpolation()), Interpolations.Flat())




# Take derivative to get V′_old_interp


## Step 2: While above some tolerance, find the new value function and update the old one.

function VFI(u, u′, u′_inv, γ, β, income_states, income_probs, return_states, return_probs, w_grid, tol)

    dist = 1 + tol
    i = 0

    grid_size = length(w_grid)

    ### Set up the initial guess for the value function.
    ### Use zeros

    V_old = u.(w_grid, γ)

    ### Form a PCHIP interpolant for V_old.

    V_old_interp = extrapolate(interpolate(w_grid, V_old, SteffenMonotonicInterpolation()), Interpolations.Flat())

    ### Also specify the derivative of V_old_interp

    V′_old = u′.(w_grid, γ)

    V′_old_interp = extrapolate(interpolate(w_grid, V′_old, SteffenMonotonicInterpolation()), Interpolations.Flat())

    while dist > tol 
    
        V_new = find_new_V(V_old_interp, V′_old_interp, u, u′_inv, γ, β, income_states, income_probs, return_states, return_probs, w_grid)

        # Calculate distance between V_new and V_old.
        i = i + 1
        println("Iteration: ", i)
        dist = deepcopy(maximum(abs.(V_new - V_old)))
        println("Distance: ", dist)
        println(" ")
        
        println("Old V")
        display(V_old)

        println("New V")
        display(V_new)

        V_new_interp = extrapolate(interpolate(w_grid, V_new, SteffenMonotonicInterpolation()), Interpolations.Flat())
        
        V_old = deepcopy(V_new)
        V_old_interp = deepcopy(V_new_interp)

        # Get derivative as well. 
        ## Calculate derivative by averaging slope to either side of each point in V_old. 

        for i in 1:grid_size
            if i == 1
                V′_old[i] = (V_old_interp(w_grid[i + 1]) - V_old_interp(w_grid[i])) / (w_grid[i + 1] - w_grid[i])
            elseif i == grid_size
                V′_old[i] = (V_old_interp(w_grid[i]) - V_old_interp(w_grid[i - 1])) / (w_grid[i] - w_grid[i - 1])
            else
                V′_old[i] = (V_old_interp(w_grid[i + 1]) - V_old_interp(w_grid[i - 1])) / (w_grid[i + 1] - w_grid[i - 1])
            end
        end

        ## Form a monotonic interpolant for V′_old.

        V′_old_interp = extrapolate(interpolate(w_grid, V′_old, SteffenMonotonicInterpolation()), Interpolations.Flat())

    end

    return V_old 

end

tol = 10^(-10)
V_new = VFI(u, u′, u′_inv, γ, β, income_states, income_probs, return_states, return_probs, w_grid, tol)

plot(w_grid[2:end], V_new[2:end], label = "Value Function")