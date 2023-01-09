# set current folder as the working directory
cd(@__DIR__)

include("Primitives.jl")
# Be sure to have run the below three scripts, since this script makes 
# use of the outputs from thos scripts stored in the /temp folder. 
# include("Q2_EGM.jl")
# include("Q3_EGM.jl")
# include("Q6_EGM.jl")

# -------------------------------------------------------------------

# (1) No Loans 

C_no_loans = load("temp/C_no_loans.jld", "C_no_loans")


#### Define a value function matrix
V = zeros(n, NTauchen, T)

#### specify that in the final period you just get the utility from consuming 
#### all cash on hand. 

for (i, a) in enumerate(a′_grid)
    for (j, x) in enumerate(x_states)
        V[i, j, T - 1] = u.(A[T - 1] * x + R * a, γ)
    end
end

#### Use the consumption function to infer previous period's value function.

for t in T-2:-1:1
    println(t)
    ## Form an interpolation to value function at t + 1
    V_interpolant = LinearInterpolation((a′_grid, x_states), V[:, :, t + 1])
    for (i, a) in enumerate(a′_grid)
        for (j, x) in enumerate(x_states)
            cons_opt = C_no_loans[i, j, t]
            next_period_assets = R * a + x_states[j] * A[t] - cons_opt
            if next_period_assets > a′_grid[end]
                next_period_assets = a′_grid[end] # only occurs at the very edge of the grid we don't care about anyway. 
            end
            V[i, j, t] = u(cons_opt, γ) + β * (transpose(Π_x[j, :]) * V_interpolant.(next_period_assets, x_states))
        end
    end
end

V_no_loans = deepcopy(V)
save("temp/V_no_loans.jld", "V_no_loans", V_no_loans)

# -------------------------------------------------------------------

# (2) Loans 

C_with_loans = load("temp/C_with_loans.jld", "C_with_loans")

#### Define a value function matrix
V = zeros(n, NTauchen, T)

#### specify that in the final period you just get the utility from consuming 
#### all cash on hand. 

for (i, a) in enumerate(a′_grid)
    for (j, x) in enumerate(x_states)
        V[i, j, T - 1] = u.(A[T - 1] * x + R * a, γ)
    end
end

#### Use the consumption function to infer previous period's value function.

for t in T-2:-1:1

    println(t)
    
    if t <= 10
        L_t = Lbar * (R_c - 1) * R_c^10 / (R_c^10 - 1)
    else
        L_t = 0
    end
    
    ## Form an interpolation to value function at t + 1
    V_interpolant = LinearInterpolation((a′_grid, x_states), V[:, :, t + 1])
    for (i, a) in enumerate(a′_grid)
        for (j, x) in enumerate(x_states)
            cons_opt = C_with_loans[i, j, t]
            next_period_assets = R * a + x_states[j] * A[t] - L_t - cons_opt
            if next_period_assets > a′_grid[end]
                next_period_assets = a′_grid[end] # only occurs at the very edge of the grid we don't care about anyway. 
            end
            ## It's possible to have rounding errors on the scale of 10^{-13} as well that can break the code but don't 
            ## matter economically.
            ## These can be ignored, but we throw an error if ever they are too large. 
            if next_period_assets < a′_grid[1]
                if next_period_assets < -1e-5
                    error("next_period_assets is negative and too large")
                else
                    next_period_assets = a′_grid[1]
                end
            end
            
            V[i, j, t] = u(cons_opt, γ) + β * (transpose(Π_x[j, :]) * V_interpolant.(next_period_assets, x_states))
        end
    end
end

V_with_loans = deepcopy(V)

save("temp/V_with_loans.jld", "V_with_loans", V_with_loans)

# -------------------------------------------------------------------

# (3) Tax 

C_with_tax = load("temp/C_with_tax.jld", "C_with_tax")

# Construct the Value function. 

#### Define a value function matrix
V = zeros(n, NTauchen, T)

#### specify that in the final period you just get the utility from consuming 
#### all cash on hand. 

for (i, a) in enumerate(a′_grid)
    for (j, x) in enumerate(x_states)
        V[i, j, T - 1] = u.(A[T - 1] * x + R * a, γ)
    end
end

#### Use the consumption function to infer previous period's value function.

for t in T-2:-1:1

    println(t)
    
    
    #### Specify τ 
    if t <= 10
        τ = τ_10
    else
        τ = 0
    end
    
    ## Form an interpolation to value function at t + 1
    V_interpolant = LinearInterpolation((a′_grid, x_states), V[:, :, t + 1])
    for (i, a) in enumerate(a′_grid)
        for (j, x) in enumerate(x_states)
            cons_opt = C_with_tax[i, j, t]
            next_period_assets = R * a + x_states[j] * A[t] * (1 - τ) - cons_opt
            if next_period_assets > a′_grid[end]
                next_period_assets = a′_grid[end] # only occurs at the very edge of the grid we don't care about anyway. 
            end
            ## It's possible to have rounding errors on the scale of 10^{-13} as well that can break the code but don't 
            ## matter economically.
            ## These can be ignored, but we throw an error if ever they are too large. 
            if next_period_assets < a′_grid[1]
                if next_period_assets < -1e-5
                    error("next_period_assets is negative and too large")
                else
                    next_period_assets = a′_grid[1]
                end
            end
            
            V[i, j, t] = u(cons_opt, γ) + β * (transpose(Π_x[j, :]) * V_interpolant.(next_period_assets, x_states))
        end
    end
end

V_with_tax = deepcopy(V)

save("temp/V_with_tax.jld", "V_with_tax", V_with_tax)





