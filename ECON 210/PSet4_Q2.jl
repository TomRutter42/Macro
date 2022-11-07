using LinearAlgebra

# Define the transition matrix
Π = [0.5 0.4 0.1; 0.3 0.4 0.3; 0.2 0.3 0.5]

# Calculate the rank of the transition matrix
println("rank(Π) = ", rank(Π))

# Calculate the identity matrix substract the transition matrix plus the matrix of ones 
A = I - Π + ones(3, 3)

# Calculate the inverse of A
Ainv = inv(A)
print(Ainv)

# Multiply the row vector of ones by the inverse of A
v = ones(1, 3) * Ainv
print(v)

# # Left eigenvalue is the eigenvalue of transpose. Hence we transpose and find the eigenvalue 
# Π′ = transpose(Π)
# π = eigen(Π′, sortby = x -> -abs(x)).vectors[:, 1] # We know that 1 will be the largest eigenvalue
# π = π / sum(π) # Normalize the eigenvector so it corresponds to a prob. distribution

# #Pick an unstable initial distribution
# π_0 = ones(3) / 3
# π_100 = transpose(π_0) * Π ^ 100

# # Print the stationary distribution and the distribution after 100 iterations
# println("The stationary distribution is: ", π)
# println("The distribution after 100 iterations is: ", π_100)

# # Check whether we have converged close enough to the invariant distribution π
# error = norm(π_100 - transpose(π))
# if error < 1e-8
#     print("Converged")
# else
#     @show "Probability is $π_s_100" 
# end

test = [1.91  -0.27 -1.02; -0.49 1.53 -0.42 ; -0.79 -0.57 1.98]
# premultiply by vector of ones 
v = ones(1, 3) * test
# divide by 1/1.86
v = v / 1.86
print(v) 