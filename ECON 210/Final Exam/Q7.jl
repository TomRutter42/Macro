cd(@__DIR__)

include("Primitives.jl")
# Be sure to run this file first. 
# include("Construct_Value_Functions.jl")

# -------------------------------------------

Expected_utility_tax = zeros(length(a′_grid))
Expected_utility_loan = zeros(length(a′_grid))
epsilon_ex_ante = zeros(length(a′_grid))
for i in 1:length(a′_grid)
    Expected_utility_tax[i] = 
        transpose(stat_dist) * V_with_tax[i, :, 1]

    Expected_utility_loan[i] = 
        transpose(stat_dist) * V_with_loans[i, :, 1]

    epsilon_ex_ante[i] = (Expected_utility_tax[i] / Expected_utility_loan[i])^(1 / (1 - γ)) - 1

end
    
xmax = 200000
xlim = 210000

plot(a′_grid, epsilon_ex_ante, 
     ylabel = "ε", xlabel = "Assets for period 0", 
     formatter = :plain, legend = :none, 
     xlims = (0, xlim), 
     xticks = 0:increment:xmax)

savefig("figures/ExAnteEpsilon.png")