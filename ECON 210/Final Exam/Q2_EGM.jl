# Solve Question 2 Using Endogenous Grid Method

# set current folder as the working directory
cd(@__DIR__)

include("Primitives.jl")
include("IncomeDiscretizationForSourcing.jl")
include("EGM_Helper_Functions.jl")

# ---------------------

### Use the budget constraint to solve for savings function
### a(b′, y), where a(b′, y) is the value of assets today that would
### lead the consumer to have b′ tomorrow, if income is y today.

### Here we are calculating the endogenous grid.
    
function calculate_savings_consumption(R, β, γ, Π_x, a′_grid, x_states, c_next, A, t)

    # Define the grid sizes
    nbp = length(a′_grid)
    ny = length(x_states)

    # Create an empty matrix to store the savings function
    a_endog_grid = zeros(nbp, ny)
    a_thr = zeros(ny)

    # Calculate the consumption function
    c_endog_grid = solve_c(R, β, γ, Π_x, a′_grid, x_states, c_next)

    y_states = x_states .* A[t]

    for (j, y) in enumerate(y_states)

        ## threshold that gets the borrowing constraint to bind next period.
        a_thr[j] = (1 / R) * (c_endog_grid[1, j] - y) 

    end

    # Loop over the grid points for b′ and y
    for (i, a′) in enumerate(a′_grid)
        for (j, y) in enumerate(y_states)

            a_endog_grid[i, j] = (1 / R) * (c_endog_grid[i, j] + a′ - y)

        end
    end

    return a_endog_grid, c_endog_grid, a_thr

end

# ---------------------

# Express consumption as function of assets today, not tomorrow. 

function c_function_of_a(R, β, γ, Π_x, a′_grid, x_states, c_next, A, t)

    y_grid = x_states .* A[t]

    # Define the grid sizes
    nbp = length(a′_grid)
    ny = length(y_grid)

    # Calculate the savings function
    a_endog_grid, c_opt, a_thr = 
        calculate_savings_consumption(R, β, γ, Π_x, a′_grid, x_states, c_next, A, t)
    ## Note that c_opt implictly gives the value of c for the endogenous grid.

    # # Create an empty matrix to store the updated consumption function
    c_new = zeros(nbp, ny)

    ## We have a matrix of consumption c_opt as a function of current
    ## income y and next period assets a′.
    ## We want instead consumption as a function of current assets a and income y. 
    ## We can do this by interpolating the consumption function c_opt.
    ## For each b′ and y, we want to linearly interpolate the consumption
    ## function c_opt between the two grid points in b_opt that bracket b′.

    # Loop over the grid points for b′ and y

    for (i, a′) in enumerate(a′_grid)
        for (j, y) in enumerate(y_grid)

            # Case 1: Liquidity Constraint is binding.

            if a′ <= a_thr[j]
                c_new[i, j] = y + R * a′ + 0.0 ## here we just use a′_grid as our a-values. 
            else
                # Case 2: Liquidity Constraint is not binding.
                # Just use the Euler equation.
                ## Find the index of the first grid point in b_opt that is greater
                ## than b′.
                ind = findfirst(a_endog_grid[:, j] .> a′)

                ## If a′ is greater than the largest grid point in a_opt, then
                ## set the index to be the last grid point in a_opt.
                if ind == nothing
                    ind = nbp
                end

                ## If b′ is less than the smallest grid point in a_opt, then
                ## set the index to be the first grid point in a_opt.
                if ind == 1
                    ind = 2
                end

                ## Linearly interpolate the consumption function c_opt between
                ## the two grid points in b_opt that bracket b′.
                c_new[i, j] = 
                    c_opt[ind - 1, j] + (c_opt[ind, j] - c_opt[ind - 1, j]) *
                    (a′ - a_endog_grid[ind - 1, j]) / 
                    (a_endog_grid[ind, j] - a_endog_grid[ind - 1, j])
            end
        end
    end

    return c_new

end


# ----------------------------------------------------------------------------- #

# Iterate back from final period to first. 

## Specify that in the final period where the agent consumes, T - 1, 
## optimal consumption is just the agent consuming all of their cash on hand. 

for (i, a) in enumerate(a′_grid)
    for (j, x) in enumerate(x_states)
        C[i, j, T - 1] = A[T - 1] * x + R * a
    end
end

for t in T-2:-1:1
    println(t)
    C[:, :, t] = c_function_of_a(R, β, γ, Π_x, a′_grid, x_states, C[:, :, t + 1], A, t)
end

# Save this consumption function for later questions where we will need it. 

C_no_loans = deepcopy(C)
## Save this consumption array to the /temp folder.
save("temp/C_no_loans.jld", "C_no_loans", C_no_loans)

# ----------------------------------------------------------------------------- #

# Plot Consumption Function

## Plot Consumption at T = 1

y_grid = x_states .* A[1]

# ## Create NTauchen separate vectors, each of which displays 
# ## the cash on hand given the value of assets a (we use a′_grid here) 
# ## plus income. 

CashOnHand = zeros(length(a′_grid), NTauchen)

for (j, y) in enumerate(y_grid)
    CashOnHand[:, j] = y .+  R * a′_grid
end

# 30 year-old consumption function. 
y_grid30 = x_states .* A[6]
CashOnHand30 = zeros(length(a′_grid), NTauchen)

for (j, y) in enumerate(y_grid30)
    CashOnHand30[:, j] = y .+ R * a′_grid
end

xlim = 210000
ymax = 110000
increment = 50000
xmax = 200000

display(y_grid[1])
display(y_grid[Int((NTauchen + 1) / 2)])
display(y_grid30[1])
display(y_grid30[Int((NTauchen + 1) / 2)])

plot(CashOnHand[:, 1], C[:, 1, 1], label = "Age 25: Lowest Income (7,000)", 
     ylabel = "Consumption", xlabel = "Cash on Hand", 
     formatter = :plain, legend = :topleft, 
     xlims = (0, xlim), 
     xticks = 0:increment:xmax)
plot!(CashOnHand30[:, 1], C[:, 1, 6], label = "Age 30: Lowest Income (9,500)")
plot!(CashOnHand[:, Int((NTauchen + 1) / 2)], C[:, Int((NTauchen + 1) / 2), 1], 
      label = "Age 25: Medium Income (50,000)")
plot!(CashOnHand30[:, Int((NTauchen + 1) / 2)], C[:, Int((NTauchen + 1) / 2), 6],
      label = "Age 30: Medium Income (67,000)")
plot!([0, ymax], [0, ymax], label = "45 Degree Line", 
      linestyle = :dot, color = :black)

savefig("figures/ConsumptionFunctionAge25vs30.png")
