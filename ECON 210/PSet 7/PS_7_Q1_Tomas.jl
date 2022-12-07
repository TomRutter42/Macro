using Distributed
@everywhere using LinearAlgebra, Distributions, Random
@everywhere using Optim, Parameters, Interpolations
@everywhere using Optim: maximum, maximizer
@everywhere using LaTeXStrings, Colors
@everywhere using QuantEcon, Plots, Pkg, Dierckx
@everywhere using Distributed


# set current folder as the working directory
cd(@__DIR__)

# Define useful subfunction
@everywhere isoelastic(c, γ) = isone(γ) ? log(c) : (c^(1 - γ) - 1) / (1 - γ)
## Discretize normal Distributions
@everywhere function Aprrox_normal(N, σ)
    Π = QuantEcon.tauchen(N, 0, σ, 0, 3)
    # States of the markov Chain
    s = Π.state_values
    Π = Π.p[1,:]
    # Simulate Markov Chain
    return s, Π
end

# Define model parameters
@everywhere Portfolio_problem = @with_kw (  β = 0.90,                                             # Discount factor
                                γ = 2,                                               # EIS inverse
                                Rf = 1.01,                                            # Risk free rate
                                σ = sqrt(0.01),                                       # income var
                                nW = 1000,                                             # Cash on hand grid size
                                nφ = 100,                                             # Portfolio grid size
                                nS = 11,                                              # State grid size
                                w_grid_min = 1e-6,                                                      # Cash on hand grid lower bound
                                w_grid_max = 500,                                                       # Cash on hand grid upper bound
                                w_grid = range(w_grid_min, w_grid_max, nW),                             # cash on hand grid
                                φ_grid_min = 0,                                                         # Portfolio on hand grid lower bound
                                φ_grid_max = 1,                                                         # Portfolio on hand grid upper bound
                                φ_grid = range(φ_grid_min, φ_grid_max, nφ),
                                #w_grid = exp.(range(0, log(w_grid_max), 
                                #length = nW)) .- 1 .+ w_grid_min,                                      # cash on hand grid
                                u = (c, γ=γ) -> isoelastic(c, γ),                                       # Utility function
                                y_Monte_carlo = exp.(rand(Normal(4.7, 0.01), 250)),                     # Monte Carlo income
                                Rr_Monte_carlo = exp.(rand(Normal(0.08, 0.11), 250)),                   # Monte Carlo risky return
                                ext_pt = w_grid_max*(Rf + maximum(Rr_Monte_carlo) ) 
                                + maximum(y_Monte_carlo),                                                # grid extension point
                                y_Tauchen = exp.(Aprrox_normal(nS, sqrt(0.01))[1] .+ 4.7),              # Tauchen discretization for income
                                Πy = Aprrox_normal(nS, sqrt(0.01))[2],                                   # Tauchen discretization for income
                                R_Tauchen = exp.(Aprrox_normal(nS, sqrt(0.11))[1] .+ 0.08),             # Tauchen discretization for return
                                ΠR = Aprrox_normal(nS, sqrt(0.01))[2]                                   # Tauchen discretization for return
                
)

# RHS of Bellman
@everywhere function RHS(V_f, c, φ, m, w) # Perhaps don't load it through model
    @unpack β, Rf, u, γ, y_Monte_carlo, Rr_Monte_carlo = m()
    R_w′ = Rf .+ φ .* (Rr_Monte_carlo .- Rf)
    w′ = R_w′.*(w .- c) .+ y_Monte_carlo
    value = u(c) .+ β .* mean(V_f.(w′))
    return value
end


@everywhere function RHS_Tauchen(V_f, c, φ, m, w) # Perhaps don't load it through model
    @unpack β, Rf, u, γ, y_Tauchen, R_Tauchen, Πy , ΠR = m()
    value = u(c) + β * sum([Πy[y_i]*ΠR[r_i]*V_f( Rf + φ * (Rr - Rf).*(w - c) + y) for (r_i, Rr) in enumerate(R_Tauchen) for (y_i, y) in enumerate(y_Tauchen)])
    return value
end

# Bellman operator
@everywhere function T(V,m,φ; tol = 1e-8)
    @unpack w_grid, ext_pt = m()

    # Linearly interpolate the value function
    #grid_extend = vcat(0.0, collect(w_grid), 2*ext_pt)    # to avoid boundary errors, we need
    #V_extend = vcat(V[1],  V,  V[end])                # to extend the range of the grid
    #V_f = LinearInterpolation(grid_extend, V_extend)  # used for the interpolation
    # Take cubic spline of V_f
    V_f = Spline1D(w_grid, V; k = 3)
    lower = [0.0, 0.0]
    # Solve Bellman equation
    TV = similar(V)
    Threads.@threads for (i, w) in collect(enumerate(w_grid))
        #Objective = (x) -> -1*RHS(V_f,x,m,w)
        upper = [1.0, w]
        initial_x = [0.5, w/2]
        #od = OnceDifferentiable((x) -> -1*RHS(V,x,m,w), initial_x)
        #results = optimize(od, initial_x, lower, upper, Fminbox{GradientDescent}())
        
        #results = optimize(OnceDifferentiable((x) -> -1*RHS(V_f,x,m,w), initial_x),initial_x, lower, upper, Fminbox())
        results = maximize(c -> RHS_Tauchen(V_f,c,φ,m,w), tol, w)
        #results = optimize((x) -> -1*RHS(V_f,x,m,w),lower, upper, initial_x)
        TV[i] = maximum(results)
    end
    return TV
end

# Value function interation
@everywhere function VFI(m,φ)
    @unpack w_grid, nW, u, β, Rf, y_Monte_carlo, Rr_Monte_carlo = m()
    V = u.(w_grid)
    TV = similar(V)
    tol = 1e-15
    dist = 1.0
    max_iter = 1000
    i = 0
    while dist > tol #|| i <= max_iter
        i += 1
        TV = T(V, m, φ)
        dist = maximum(abs.(TV .- V))
        V = copy(TV)
        print(dist)
    end
    return TV
end

V = zeros(Portfolio_problem().nW,Portfolio_problem().nφ)
# measure time
Threads.@threads for (φ_i,φ) in collect(enumerate(Portfolio_problem().φ_grid))
    V[:,φ_i] = VFI(Portfolio_problem,φ)
end

