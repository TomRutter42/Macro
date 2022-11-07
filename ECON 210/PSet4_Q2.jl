## Code to solve the problem in question 2 of PSet 4

using LinearAlgebra

# Define the transition matrix
Π = [0.5 0.4 0.1; 0.3 0.4 0.3; 0.2 0.3 0.5]

# -------------------------------------------------------------------

# Part (b) 
# Calculate the stationary distribution of the Markov chain

## Left eigenvalue is the eigenvalue of transpose. 
## Hence we transpose and find the eigenvalue 
Π′ = transpose(Π)
### We know that 1 will be the largest eigenvalue, so select based on this. 
π = eigen(Π′, sortby = x -> -abs(x)).vectors[:, 1] 
### Normalize the eigenvector so it corresponds to a prob. distribution
π = π / sum(π) 

println("The stationary distribution is: ", π)

# -------------------------------------------------------------------

# Part (c)
# Pick an unstable initial distribution
π_0 = ones(3) / 3

# -------------------------------------------------------------------

# Part (d)

## Calculate the distribution tomorrow

π_1 = transpose(π_0) * Π
println("The distribution tomorrow is: ", π_1)

## Calculate the distribution the day after tomorrow 

π_2 = transpose(π_0) * Π ^ 2
println("The distribution the day after tomorrow is: ", π_2)

## Calculate the distribution after 100 periods 

π_100 = transpose(π_0) * Π ^ 100
println("The distribution after 100 periods is: ", π_100)

## Check whether we have converged close enough to the invariant distribution π
error = norm(π_100 - transpose(π))
if error < 1e-8
    println("Process converged")
else
    println("Process did not converge")
end

## Print the stationary distribution and the distribution after 100 iterations
println("The stationary distribution is: ", π)
println("The distribution after 100 iterations is: ", π_100)

# -------------------------------------------------------------------


