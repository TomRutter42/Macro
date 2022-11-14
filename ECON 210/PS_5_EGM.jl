# Solve the 1991 Deaton consumption-savings problem 
# using the endogenous grid method. 

# ----------------------------------------------------------------------------- #

# Packages 
using Distributions
using Plots
using QuantEcon

# ============================================================================= #

# Define Primitives:

## Utility function

function u(c, γ)
    if c <= 0.01 
        return -100000
    elseif γ == 1
        return log(c)
    else
        return c^(1 - γ) / (1 - γ)
    end
end

## Derivative of utility function

function u′(c, γ)
    if c <= 0.01 
        return 1000
    elseif γ == 1
        return 1/c
    else
        return c^(-γ)
    end
end

## Inverse of derivative of utility function

function u′_inv(MU, γ)
    return (MU)^(-1 / γ)
end

# ============================================================================= #

# Parameters 
r = 0.05 
δ = 0.10 
ρ = 2 

# ============================================================================= #

# EGM 

# ----------------------------------------------------------------------------- #

## Step 1: Define the transition probabilities for the income process, 
### given the persistence parameter of the AR(1) ψ, mean μ, and 
### standard deviation σ, using the Tauchen method.

### Construct transition matrix for the AR(1) process. 
ny = 200 # number of grid points for income y
MChain = QuantEcon.tauchen(ny, ψ, σ, μ, 3)
Π = MChain.p
y_grid = MChain.state_values

# ----------------------------------------------------------------------------- #

## Step 2: Construct a grid on (b′, y) 

### Define the grid sizes. 
nbp = 200 # number of grid points for bond choice b′ 

### Define the grids.
### Use a log-spaced grid for saving b′.
### Note that the saving grid implicitly contains the borrowing constraint.
b′_grid = log(range(1.0, exp(300), length = nbp)) 

# ----------------------------------------------------------------------------- #

## Step 3: Form a guess for the consumption function 

### Define the guess for the consumption function.
c_old = zeros(nbp, ny)
### Fill in c_old with r * b′ + y. 
for i in 1:nbp 
    for j in 1:ny 
        c_old[i, j] = r * b′_grid[i] + y_grid[j]
    end
end

# ----------------------------------------------------------------------------- #

## Step 4: Calculate RHS of Euler equation. 

### Create a function to calculate the RHS of the Euler equation. 

function calculate_RHS(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Define the grid sizes
    nbp = length(b′_grid)
    ny = length(y_grid)

    # Create an empty matrix to store the RHS of the Euler equation
    RHS = zeros(nbp, ny)

    # Loop over the grid points for b′ and y
    for (i, b′) in enumerate(b′_grid)
        for (j, y) in enumerate(y_grid)
            for (k, y′) in enumerate(y_grid)
                ## Calculate utility for each realization of y′ and take expectation.
                RHS[i, j] += ((1 + r) / (1 + \delta)) * Π[j, k] * u′(c_old[i, k], ρ)
            end
        end
    end

    return RHS

end


# ----------------------------------------------------------------------------- #

## Step 5: Solve for consumption satisfying Euler equation.

### Create a function to solve for the consumption function for b′.

### Note this is the heart of the speedup---here we simply solve for c 
### analytically rather than solving for a root numerically. 

function solve_c(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Define the grid sizes
    nbp = length(b′_grid)
    ny = length(y_grid)

    # Create an empty matrix to store the consumption function
    c_new = zeros(nbp, ny)

    # Calculate the RHS of the Euler equation
    RHS = calculate_RHS(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Loop over the grid points for b′ and y
    for (i, b′) in enumerate(b′_grid)
        for (j, y) in enumerate(y_grid)
            c_new[i, j] = u′_inv(RHS[i, j], ρ)
        end
    end

    return c_new

end

# ----------------------------------------------------------------------------- #

## Step 6: Use the budget constraint to calculate the savings function.

### Create a function to calculate the savings function.

### Here we are calculating the endogenous grid. 

function calculate_savings_consumption(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Define the grid sizes
    nbp = length(b′_grid)
    ny = length(y_grid)

    # Create an empty matrix to store the savings function
    b_opt = zeros(nbp, ny)

    # Calculate the consumption function
    c_new = solve_c(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Loop over the grid points for b′ and y
    for (i, b′) in enumerate(b′_grid)
        for (j, y) in enumerate(y_grid)
            b_opt[i, j] = (1 / (1 + r)) * (c_new[i, j] - y_grid[j] + b′_grid[i])
        end
    end

    return b_opt

end

# ----------------------------------------------------------------------------- #

## Update the guess for the consumption function.

### Create a function to update the guess for the consumption function.

function update_c(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Define the grid sizes
    nbp = length(b′_grid)
    ny = length(y_grid)

    # Calculate the savings function
    b_opt = calculate_savings(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Create an empty matrix to store the updated consumption function
    c_new = zeros(nbp, ny)

    # Loop over the grid points for b′ and y
    for (i, b′) in enumerate(b′_grid)
        for (j, y) in enumerate(y_grid)
            ## fill me in 
        end
    end

    return c_new

end

# THEN NEED TO REPEAT ALL OF THE ABOVE UNTIL YOU GET CONVERGENCE







