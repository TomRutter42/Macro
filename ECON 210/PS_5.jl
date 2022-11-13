using QuantEcon, Plots, Pkg, Dierckx, NLsolve, LinearAlgebra

# set current folder as the working directory
cd(@__DIR__)


# Specify the AR(1) paramaters on the basis of answer to previous questions
φ = 0.7
σ = 10
 #3.114 is the mean inflation
Ey = 100
μ_w = Ey*(1-φ)
# Discretize the AR(1) using Tauchen method
function Aprrox_MC(N, φ, σ, μ,bounds)
    Π = QuantEcon.tauchen(N, φ, σ, 0,bounds)
    # States of the markov Chain
    s = Π.state_values .+ μ
    Π = Π.p
    # Simulate Markov Chain
    return s, Π
end

# Obtain particular discretization for the problem
nS = 10
(s, Π) = Aprrox_MC(nS, φ, σ, Ey,3)

# Risk aversion from Deaton
ρ = 2
β = 0.95
R = 1.05

# Define CRRA utility function
function u(c, γ)
    if γ == 1
        return log(c)
    else
        return (c^(1-γ) - 1)/(1-γ)
    end
end

# Define the derivative of the utility function
function u′(c, γ)
    if γ == 1
        return 1/c
    else
        return c^(-γ)
    end
end


# Define grids for cash-on-hand and bond holdings
nW = 1000
nB = 1000
w_grid = range(0.01,300,nW) # Grid on cash on hand
# note that the lower bound on w_grid can be obtained from the fact that w = b + y
b_grid = range(0.01,300,nB) # Grid on borrowing

# Initialize policy function
b′ = zeros(nW, nS)
w′ = zeros(nW, nS)
c = zeros(nW, nS)

# Initialize value function
V = zeros(nW,nS)
# Reasonable guess: for every element of the grid, the value function is the utility of the grid element
for (i_w, w) in enumerate(w_grid)
    for j in 1:nS
        V[i_w,j] = u(w, ρ)
    end
end
# Initialize policy function
b′_sol = zeros(nW,nS)
w′_sol = zeros(nW,nS)
c_sol = zeros(nW,nS)

# Define Bellman operator
function T!(V)
    VV_new = copy(V)
    VV_cands = zeros(nB)
    # Define an array of splines of V
    V_splines = [Spline1D(w_grid, V[:,i]; k = 3) for i in 1:nS]
    for (w_index, w)  in enumerate(w_grid)
        for (y_index, y) in enumerate(s)
            for (b′_index, b′) in enumerate(b_grid) # Borrowing constraint is imposed from the grid.
                if w <= b′/R
                    VV_cands[b′_index] = -1e30
                else
                    rhs_bellman = u(w - b′/R, ρ) + β*sum(V_splines[y′_index](b′ + y′)*Π[y_index,y′_index] for (y′_index, y′) in enumerate(s))
                    # for (y′_index, y′) in enumerate(s)
                        # w′ = b′ + y′
                        # w′_index = findfirst(x -> x >= w′, w_grid) # Find the index of the first element of w that is greater than wp
                        # Marcelo used findmin
                        # wp_pos = findmin(abs.(w - wp))[2] # Find the index of the element of w that is closest to wp
                    #    rhs_bellman += β*V_splines[y′_index](w′)*Π[y_index,y′_index]
                    # end
                    VV_cands[b′_index] = rhs_bellman
                end
            end
            VV_new[w_index,y_index] = maximum(VV_cands)
        end
    end
    return VV_new
end



T!(V) # Check that the Bellman operator works




# Value function iteration
tol = 1e-6 # Tolerance for convergence
max_iter = 1000 # Maximum number of iterations
V_sol = fixedpoint(T!, V, iterations=max_iter, xtol=tol, m = 0).zero

# Find the policy functions
V_sol_Spline = [Spline1D(w_grid, V_sol[:,i]; k = 3) for i in 1:nS]
for (w_index, w)  in enumerate(w_grid)
    for (y_index, y) in enumerate(s)
        # Find the index of the optimal bond holdings
        VV_cands = zeros(nB)
        for (b′_index, b′) in enumerate(b_grid) # Borrowing constraint is imposed from the grid.
            if w <= b′/R
                VV_cands[b′_index] = -1e30
            else
                rhs_bellman = u(w - b′/R, ρ) + β*sum(V_sol_Spline[y′_index](b′ + y′)*Π[y_index,y′_index] for (y′_index, y′) in enumerate(s))
                VV_cands[b′_index] = rhs_bellman
            end
        end
        b′_sol[w_index,y_index] = b_grid[argmax(VV_cands)]
        # DOES NOT WORK: b′_sol[w_index,s] = argmax(b′ -> u(w - R^-1 * b′, ρ) + β*sum(V_sol_Spline[y′_index](b′ + y′)*Π[s,y′_index] for (y′_index, y′) in enumerate(s)))
        # Find the optimal cash-on-hand
        c_sol[w_index,y_index] = w - R^-1 * b′_sol[w_index,y_index]
    end
end
# Plot the policy functions with axes having ticks of the same size
plot(w_grid[1:40], b′_sol[1:40,5], label = "b′", title = "Policy functions", xlabel = "w", ylabel = "b′")
plot(w_grid, c_sol[:,3], label = "c", xlabel = "w", ylabel = "c", title = "Policy function for consumption")
plot!(w_grid, w_grid, label = "45 deg line")
plot(w_grid, w_grid .- c_sol[:,2], label = "c", xlabel = "w", ylabel = "c", title = "Policy function for consumption")

# Plot the c so that each column of c has different colour
plot(w_grid, c_sol, label = "c", xlabel = "w", ylabel = "c", title = "Policy function for consumption", legend = :topleft)




(w_grid .- c_sol[:,2] .>= 0

# Iterate the value function until convergence#

# Initialize value function
V = zeros(nW,nS)
# Reasonable guess: for every element of the grid, the value function is the utility of the grid element
for (i_w, w) in enumerate(w_grid)
    for j in 1:nS
        V[i_w,j] = u(w, ρ)
    end
end
tol = 1e-6 # Tolerance for convergence
max_iter = 1000 # Maximum number of iterations
error = 10
i = 0
while i < max_iter && error > tol
        V_next = 0.5*V +0.5*T!( V)
        error = norm(V_next - V)
        i += 1
        V = V_next  # copy contents into V.  Also could have used V[:] = v_next

    if i == max_iter
        error("Didn't converge")      
    else
        for (w_index, w)  in enumerate(w_grid)
            for (s, y) in enumerate(s)
                # Find the index of the optimal bond holdings
                b′_sol[w_index,s] = argmax(b′ -> u(w - R^-1*b′, ρ) + β*sum(Π[s,:].*V_sol[s,:]), b′)
                # Find the optimal cash-on-hand
                c_sol[w_index,s] = w - R^-1 * b′_sol[w_index,s]
            end
        end
    end
end




# Plot the policy functions


argmax(b -> u(w[5] - b, ρ) + β*sum(Π[3,:].*V[5,:]), b)