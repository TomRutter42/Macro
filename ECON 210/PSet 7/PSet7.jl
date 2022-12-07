# Solve Q1 of PSet 7. 

# set current folder as the working directory
cd(@__DIR__)

# ----------------------------------------------------------------------------- #

# Packages
using Interpolations
using Plots
using QuantEcon

# ============================================================================= #

# Define Primitives:

## Utility function

function u(c, γ)
    if c <= 0.01
        return -100000
    # elseif γ == 1
    #     return log(c)
    else
        return c^(1 - γ) / (1 - γ)
    end
end

## Derivative of utility function

function u′(c, γ)
    if c <= 0.01
        return 1000
    # elseif γ == 1
    #     return 1/c
    else
        return c^(-γ)
    end
end

## Inverse of derivative of utility function

function u′_inv(MU, γ)
    return (MU)^(-1 / γ)
end

γ = 10 # risk aversion
β = 0.9 
μ_x = 4.7 
σ_x = sqrt(0.01)
R_f = 1.01 
μ_r = 0.08 
σ_r = sqrt(0.11)

# ============================================================================= #

grid_size = 100

# Set up the fixed grid for future assets.

w_grid = exp.(range(0, log(1000), length = grid_size)) .- 1

# Set up the endogenous grid for consumption. 

c_grid = exp.(range(0, log(1000), length = grid_size)) .- 1

# Set up the grid for portfolio choice ϕ

ϕ_grid = range(0, 1, length = grid_size)

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

# initial values 

w_0 = 0 
y_0 = 0 

# ============================================================================= #

# Define an initial value function.

V = zeros(grid_size)

# ============================================================================= #

# Define a function to calculate wealth distributions in the next period. 

function wealth_states(ϕ, w, c, R_f, income_states, income_probs, return_states, return_probs)

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

# Define a function that takes the possible wealth values next period, 
# applies (interpolated) V to them, and then returns the value of the RHS. 

function RHS(w′, num_income_states, num_return_states, income_probs, return_probs, V, β)

    # Note w′ is a matrix. income_probs and return_probs are vectors. 
    # V is a function. 

    ## Step 1: Apply V to each element of the matrix w′. 

    V_w′ = zeros(num_income_states, num_return_states)

    for i in 1:num_income_states
        for j in 1:num_return_states
            V_w′[i, j] = V(w′[i, j])
        end
    end

    # Run the expectation operator as a series of matrix multiplications, 
    # returning a scalar. 



    EV_w′ = transpose(income_probs) * V_w′ * return_probs

    # Confirm that EV_w′ is a scalar, returning an error message if not. 
    # println(income_probs)
    # println(return_probs)
    # println(V_w′)
    # println(EV_w′)

    # if size(EV_w′) != (1, 1)
    #     error("EV_w′ is not a scalar.")
    # end

    return β * EV_w′

end

# ============================================================================= #

# Define a function that takes a given V and returns the optimal c and ϕ for each w in w_grid.

function optimal_policy(V, u, w_grid, c_grid, ϕ_grid, R_f, num_income_states, num_return_states, income_states, income_probs, return_states, return_probs, β)

    # Initialize the policy functions. 

    c_policy = zeros(grid_size)
    ϕ_policy = zeros(grid_size)

    # Loop over the grid of w values. 

    for i in 1:grid_size

        # Loop over possible values of c and ϕ.

        # Set utility to beat to a very low number.

        best_V = -100000

        opt_j = 0
        opt_k = 0

        for j in 1:grid_size

            # if c is greater than wealth, liquidity constraint binds.

            if c_grid[j] > w_grid[i]
                continue
            end

            for k in 1:grid_size

                # Calculate the possible wealth states next period. 

                w′ = wealth_states(ϕ_grid[k], w_grid[i], c_grid[j], R_f, income_states, income_probs, return_states, return_probs)

                # Calculate the RHS of the Euler equation. 

                rh = RHS(w′, num_income_states, num_return_states, income_probs, return_probs, V, β)

                # if size(rh) != (1, 1)
                #     error("rh is not a scalar.")
                # end

                # Calculate instantaneous utility

                U = u(c_grid[j], γ)

                if size(U) != (1, 1)
                    error("rh is not a scalar.")
                end

                # Calculate total utility 

                U_total = U + rh

                # Calculate the optimal c and ϕ. 

                if U_total > best_V
                    opt_j = j
                    opt_k = k
                end

            end

        end

        # Update the policy functions.

        c_policy[i] = c_grid[opt_j]
        ϕ_policy[i] = ϕ_grid[opt_k]

    end

    return c_policy, ϕ_policy

end

# ============================================================================= #

# Define a function that takes a given V and calculates an updated estimate of V.
# This function is used in the value function iteration.


function update_V(V, u, w_grid, c_grid, ϕ_grid, R_f, num_income_states, num_return_states, income_states, income_probs, return_states, return_probs, β)

    V_new = zeros(grid_size)

    # Loop over the grid of w values. 

    for i in 1:grid_size

        # Loop over possible values of c and ϕ.

        # Set utility to beat to a very low number.

        best_V = -100000

        for j in 1:grid_size

            if c_grid[j] > w_grid[i]
                continue
            end

            for k in 1:grid_size

                # Calculate the possible wealth states next period. 

                w′ = wealth_states(ϕ_grid[k], w_grid[i], c_grid[i], R_f, income_states, income_probs, return_states, return_probs)

                # Calculate the RHS of the Euler equation. 

                rh = RHS(w′, num_income_states, num_return_states, income_probs, return_probs, V, β)

                # Calculate instantaneous utility

                U = u(c_grid[i], γ)

                # Calculate total utility 

                U_total = U + rh

                # Calculate the optimal c and ϕ. 

                if U_total > best_V 
                    best_V = U_total
                end

            end

        end

        # Update the value function.

        V_new[i] = best_V

    end

    return V_new

end

# ============================================================================= #

# Run the value function iteration process. 

# Set a tolerance level.

tol = 0.0001


## Initialize the value function.

V_init = u.(w_grid, γ)

function VFI(tol, V_init, u, w_grid, c_grid, ϕ_grid, R_f, num_income_states, num_return_states, income_states, income_probs, return_states, return_probs, β)

    dist = tol + 1

    V_spline = linear_interpolation(w_grid, V_init)
    V = deepcopy(V_init)
    i = 0
    
    while dist > tol 

        V_new = update_V(V_spline, u, w_grid, c_grid, ϕ_grid, R_f, num_income_states, num_return_states, income_states, income_probs, return_states, return_probs, β)

        # Calculate the distance between the old and new value functions. 

        dist = maximum(abs.(V_new - V))

        # Update the value function. 

        V = deepcopy(V_new)
        V_spline = linear_interpolation(w_grid, V)

        
        i += 1
        println("Iteration: ", i)
        println("Distance ", dist)
        println(V)

    end

    return V

end

VFI(tol, V_init, u, w_grid, c_grid, ϕ_grid, R_f, num_income_states, num_return_states, income_states, income_probs, return_states, return_probs, β)


