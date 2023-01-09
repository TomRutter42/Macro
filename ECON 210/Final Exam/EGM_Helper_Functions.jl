### Create a function to calculate the RHS of the Euler equation.

function calculate_RHS(R, β, γ, Π_x, a′_grid, x_states, c_next)

    # Define the grid sizes
    nbp = length(a′_grid)
    ny = length(x_states)

    # Create an empty matrix to store the RHS of the Euler equation
    RHS = zeros(nbp, ny)

    # Loop over the grid points for b′ and y
    for (i, a′) in enumerate(a′_grid)
        for (j, y) in enumerate(x_states) 
            for (k, y′) in enumerate(x_states)

                ## Calculate utility for each realization of y′ and take expectation.
                RHS[i, j] += β * R * Π_x[j, k] * u′(c_next[i, k], γ)

            end
        end
    end

    return RHS

end

# ==========================================================

### Create a function to solve for the consumption function for a′.

### Note this is the heart of the speedup---here we simply solve for c
### analytically rather than solving for a root numerically.

function solve_c(R, β, γ, Π_x, a′_grid, x_states, c_next)

    # Define the grid sizes
    nbp = length(a′_grid)
    ny = length(x_states)

    # Create an empty matrix to store the consumption function
    c_prior = zeros(nbp, ny)

    # Calculate the RHS of the Euler equation
    RHS = calculate_RHS(R, β, γ, Π_x, a′_grid, x_states, c_next)

    # Loop over the grid points for b′ and y
    for (i, a′) in enumerate(a′_grid)
        for (j, x) in enumerate(x_states)

            c_prior[i, j] = u′_inv(RHS[i, j], γ)

        end
    end

    return c_prior

end
