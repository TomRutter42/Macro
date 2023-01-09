cd(@__DIR__)

include("Primitives.jl")
# Be sure to run this file first. 
# include("Construct_Value_Functions.jl")

# -------------------------------------------

V_with_loans = load("temp/V_with_loans.jld", "V_with_loans")
V_with_tax = load("temp/V_with_tax.jld", "V_with_tax")

V = V_with_loans
V_hat = V_with_tax

epsilon = (V_hat ./ V).^(1 / (1 - γ)) .- 1

## Plot epsilon by age, wealth, and income. 

### First, plot epsilon[:, :, 1] on a grid with cash-on-Hand
### on the x-axis and the epsilon graphs split by income.

y_grid = x_states .* A[1]

# ## Create NTauchen separate vectors, each of which displays 
# ## the cash on hand given the value of assets a (we use a′_grid here) 
# ## plus income. 

CashOnHand = zeros(length(a′_grid), NTauchen)

for (j, y) in enumerate(y_grid)
    CashOnHand[:, j] = y .+  R * a′_grid
end

xlim = 410000
ymax = 110000
increment = 50000
xmax = 400000

plot(CashOnHand[:, 1], epsilon[:, 1, 1], label = "Age 25: Lowest Income (7,000)", 
     ylabel = "ε", xlabel = "Cash on Hand", 
     formatter = :plain, legend = :topright, 
     xlims = (0, xlim), 
     xticks = 0:increment:xmax)
plot!(CashOnHand[:, Int((NTauchen + 1) / 2)], epsilon[:, Int((NTauchen + 1) / 2), 1], 
      label = "Age 25: Medium Income (50,000)")
plot!(CashOnHand[:, NTauchen], epsilon[:, NTauchen, 1], 
      label = "Age 25: Medium Income (355,000)")

savefig("figures/EpsilonAge25_TAX_VS_LOAN.png")