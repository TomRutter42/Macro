# Plot Gourinchas-Parker

# set current folder as the working directory
cd(@__DIR__)

using Plots

# import V.jld and C.jld 

using JLD
V = load("V.jld", "V")
C = load("C.jld", "C")
w_hat_grid = load("w_hat_grid.jld", "w_hat_grid")

# T = number of periods
T = size(C)[2]

# Plot C at various ages

## Add 19 columns of zeros to the top of C, so that C's column index corresponds to age 
C = [zeros(size(C)[1], 19) C]

## Plot, with "Age = 26" as label 
## Restrict x-axis to be 0 to 3



plot(w_hat_grid, C[:, 26], xlabel = "Normalized Cash-On-Hand", 
     ylabel = "Normalized Consumption", label = "Age = 26", 
     legend = :topleft, xlims = (0, 3))

# loop over values, 35, 45, 55, 65 

for i = 35:10:65
    plot!(w_hat_grid, C[:, i], xlabel = "Normalized Cash-On-Hand", 
          ylabel = "Normalized Consumption", label = "Age = $i", 
          legend = :topleft, xlims = (0, 3))
end

## Save file: 
savefig("Gourinchas_Parker_C.png")

A = zeros(T)

for t = 1:T
    A[t] = (-3.12 + 0.26 * (t + 20) - 0.0024 * (t + 20)^2 ) / 1.12
end

## Add 19 zeroes to the front of the vector A, so that A's index corresponds to Age
A = [zeros(19); A]

plot(w_hat_grid ./ A[26], C[:, 26] ./ A[26], xlabel = "Normalized Cash-On-Hand", 
     ylabel = "Normalized Consumption", label = "Age = 26", 
     legend = :topleft, xlims = (0, 3))

# loop over values, 35, 45, 55, 65

for i = 35:10:65
    plot!(w_hat_grid ./ A[i], C[:, i] ./ A[i], xlabel = "Normalized Cash-On-Hand", 
          ylabel = "Normalized Consumption", label = "Age = $i", 
          legend = :topleft, xlims = (0, 3))
end

## Save file: 
savefig("Gourinchas_Parker_C_div_A.png")