using QuantEcon
using Plots

# Simulate 100 draws from income_chain 

income_draws = rand(income_chain, 100)