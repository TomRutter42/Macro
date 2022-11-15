# Solve the 1991 Deaton consumption-savings problem 
# using the endogenous grid method. 

# set current folder as the working directory
cd(@__DIR__)

# ----------------------------------------------------------------------------- #
println(" ")
println("-------------------------------------------------------------------")
println(" ")
println("Start of Run")
println(" ")

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
ny = 50 # number of grid points for income y
ψ = 0 # persistence parameter of the AR(1) process
σ = 10.0 # standard deviation of the income process
μ = 100.0 # mean of the income process
MChain = QuantEcon.tauchen(ny, ψ, σ, 0, 4.5)
Π = MChain.p
y_grid = MChain.state_values .+ μ

println("The transition matrix Π is: ")
display(Π)
println(" ")
println("The grid for y is: ")
display(y_grid)

# ----------------------------------------------------------------------------- #

## Step 2: Construct a grid on (b′, y) 

### Define the grid sizes. 
nbp = 500 # number of grid points for bond choice b′ 

### Define the grids.
### Use a log-spaced grid for saving b′.
### Note that the saving grid implicitly contains the borrowing constraint.
b′_grid = exp.(range(0, log(300), length = nbp)) .- 1
println(" ")
println("The grid for b′ is: ")
display(b′_grid)

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

println(" ")
println("The initial guess for the consumption function is: ")
display(c_old)

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
                RHS[i, j] += ((1 + r) / (1 + δ)) * Π[j, k] * u′(c_old[i, k], ρ)
            end
        end
    end

    return RHS

end

println(" ")
println("The RHS of the Euler equation, for our initial c guess, is: ")
display(calculate_RHS(r, δ, ρ, Π, b′_grid, y_grid, c_old))


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

println(" ")
println("The consumption values satisfying Euler, for our initial c guess, are: ")
display(solve_c(r, δ, ρ, Π, b′_grid, y_grid, c_old))

# ----------------------------------------------------------------------------- #

## Step 6: Use the budget constraint to solve for savings function 
## b(b′, y), where b(b′, y) is the value of assets today that would 
## lead the consumer to have b′ tomorrow, if income is y today. 

### Here we are calculating the endogenous grid. 

function calculate_savings_consumption(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Define the grid sizes
    nbp = length(b′_grid)
    ny = length(y_grid)

    # Create an empty matrix to store the savings function
    b_endog_grid = zeros(nbp, ny)
    b_thr = zeros(nbp)

    # Calculate the consumption function
    c_endog_grid = solve_c(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Loop over the grid points for b′ and y
    for (i, b′) in enumerate(b′_grid)
        for (j, y) in enumerate(y_grid)
            b_endog_grid[i, j] = (1 / (1 + r)) * (c_endog_grid[i, j] + b′_grid[i] - y_grid[j])
            ## threshold that gets the borrowing constraint to bind next period. 
            b_thr[j] = (1 / (1 + r)) * (c_endog_grid[1, j] - y_grid[j])
        end
    end

    return b_endog_grid, c_endog_grid, b_thr

end

# println(" ")
# println("The savings function, for our initial c guess, is: ")
# display(calculate_savings(r, δ, ρ, Π, b′_grid, y_grid, c_old)[1])
# println(" ")
# println("The consumption function, for our initial c guess, is: ")
# display(calculate_savings(r, δ, ρ, Π, b′_grid, y_grid, c_old)[2])
# println(" ")
# println("The threshold savings function, for our initial c guess, is: ")
# display(calculate_savings(r, δ, ρ, Π, b′_grid, y_grid, c_old)[3])

# ----------------------------------------------------------------------------- #

## Update the guess for the consumption function.

### Create a function to update the guess for the consumption function.

function update_c(r, δ, ρ, Π, b′_grid, y_grid, c_old)

    # Define the grid sizes
    nbp = length(b′_grid)
    ny = length(y_grid)

    # Calculate the savings function
    b_endog_grid, c_opt, b_thr = calculate_savings_consumption(r, δ, ρ, Π, b′_grid, y_grid, c_old)
    ## Note that c_opt implictly gives the value of c for the endogenous grid.

    # # Create an empty matrix to store the updated consumption function
    c_new = zeros(nbp, ny)

    ## We have a matrix of current assets b_opt as a function of current 
    ## income y and next period assets b′. 
    ## We want to update the consumption function, which gives a level of 
    ## consumption corresponding to bonds tomorrow b′ and current income y. 
    ## We can do this by interpolating the consumption function c_opt. 
    ## For each b′ and y, we want to linearly interpolate the consumption
    ## function c_opt between the two grid points in b_opt that bracket b′. 

    # Loop over the grid points for b′ and y

    for (i, b′) in enumerate(b′_grid)
        for (j, y) in enumerate(y_grid)

            # Case 1: Liquidity Constraint is binding.
            
            if b′ <= b_thr[j]
                c_new[i, j] = y + (1 + r) * b′ + 0
            else 
                # Case 2: Liquidity Constraint is not binding.
                # Just use the Euler equation. 
                ## Find the index of the first grid point in b_opt that is greater 
                ## than b′. 
                ind = findfirst(b_endog_grid[:, j] .> b′)

                ## If b′ is greater than the largest grid point in b_opt, then 
                ## set the index to be the last grid point in b_opt. 
                if ind == nothing
                    ind = nbp
                end

                ## If b′ is less than the smallest grid point in b_opt, then 
                ## set the index to be the first grid point in b_opt. 
                if ind == 1
                    ind = 2
                end

                ## Linearly interpolate the consumption function c_opt between 
                ## the two grid points in b_opt that bracket b′. 
                c_new[i, j] = c_opt[ind - 1, j] + (c_opt[ind, j] - c_opt[ind - 1, j]) * 
                    (b′ - b_endog_grid[ind - 1, j]) / (b_endog_grid[ind, j] - b_endog_grid[ind - 1, j])
            end
        end
    end

    return c_new

end

println(" ")
println("The updated consumption function, for our initial c guess, is: ")
display(update_c(r, δ, ρ, Π, b′_grid, y_grid, c_old))

# ----------------------------------------------------------------------------- #

## Step 7: Iterate the above until convergence. 

### Create a function to iterate the above until convergence.

function iterate(r, δ, ρ, Π, b′_grid, y_grid, c_old, tol)

    # Define the grid sizes
    nbp = length(b′_grid)
    ny = length(y_grid)

    # Create an empty matrix to store the updated consumption function
    c_new = zeros(nbp, ny)

    # Create an empty matrix to store the distance between the old and new 
    # consumption functions
    dist = zeros(nbp, ny)

    # Create an empty matrix to store the maximum distance between the old 
    # and new consumption functions
    max_dist = 1

    # Create an empty matrix to store the iteration number
    iter = 1

    # Iterate until convergence
    while max_dist > tol && iter < 2000

        # Update the guess for the consumption function
        c_new = update_c(r, δ, ρ, Π, b′_grid, y_grid, c_old)

        # Calculate the distance between the old and new consumption functions
        dist = abs.(c_new - c_old)

        # Calculate the maximum distance between the old and new consumption 
        # functions
        max_dist = maximum(dist)

        # println("Iteration number: ", iter)
        # println("Distance: ", max_dist)
        # println("The updated consumption function is: ")
        # display(c_new)
        # println("The Matrix of Distance is: ")
        # display(c_new - c_old)

        # Update the old consumption function
        c_old = deepcopy(c_new)

        

        # Update the iteration number
        iter += 1

        

    end

    return c_new

end

## Apply function 

consumption_fn = iterate(r, δ, ρ, Π, b′_grid, y_grid, c_old, 10^(-6))

println(" ")
println("The consumption function, after iterating, is: ")
display(consumption_fn)

## Plot the consumption function for each level of bonds b. 

## Plot consumption_fn when income = 70 

## then cash on hand = 70 + b 

## doesn't matter here since income is iid. 

plot(b′_grid .+ 70, consumption_fn[:, 1])
## add line y = x
# plot!(b′_grid .+ 70, b′_grid .+ 70, label = "45 degree line")

# ----------------------------------------------------------------------------- #

# Solve the Consumption savings case problem for the following four cases: 
## 1) ρ = 2, σ = 10 
## 2) ρ = 2, σ = 15 
## 3) ρ = 3, σ = 10
## 4) ρ = 3, σ = 15

consumption_fn1 = iterate(r, δ, 2.0, Π, b′_grid, y_grid, c_old, 10^(-6))
consumption_fn3 = iterate(r, δ, 3.0, Π, b′_grid, y_grid, c_old, 10^(-6))

## Small note: I tactically chose the standard deviations to calculate out 
## to in the discretization so that both y's start from 55, i.e., 
## first column of consumption function corresponds to 55 income + bonds. 

σ = 15.0 # standard deviation of the income process
MChain = QuantEcon.tauchen(ny, ψ, σ, 0, 3.0)
Π = MChain.p
y_grid = MChain.state_values .+ μ

consumption_fn2 = iterate(r, δ, 2.0, Π, b′_grid, y_grid, c_old, 10^(-6))
consumption_fn4 = iterate(r, δ, 3.0, Π, b′_grid, y_grid, c_old, 10^(-6))

## Plot each of these four cases on the same graph.

plot(b′_grid .+ 55, consumption_fn1[:, 1], label = "ρ = 2, σ = 10")
plot!(b′_grid .+ 55, consumption_fn2[:, 1], label = "ρ = 2, σ = 15")
plot!(b′_grid .+ 55, consumption_fn3[:, 1], label = "ρ = 3, σ = 10")
plot!(b′_grid .+ 55, consumption_fn4[:, 1], label = "ρ = 3, σ = 15")

# ## Add a dotted 45 degree line 
# plot!(b′_grid .+ 55, b′_grid .+ 55, label = "45 Degree Line", ls = :dot)

# Label the y axis as "Consumption"
ylabel!("Consumption")
# Label the x axis as "Cash on Hand"
xlabel!("Cash on Hand")

# Move the legend to the top left corner 
plot!(legend = :topleft)

# Restrict the axis to be 70 to 320 

plot!(xlim = (55, 320), ylim = (50, 150))

# Save the graph 
savefig("ConsumptionCases.png")

# ----------------------------------------------------------------------------- #











