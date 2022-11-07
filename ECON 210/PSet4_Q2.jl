using LinearAlgebra

# Define the transition matrix
Π = [0.5 0.4 0.1; 0.3 0.4 0.3; 0.2 0.3 0.5]

# Left eigenvalue is the eigenvalue of transpose. Hence we transpose and find the eigenvalue 
Π′ = transpose(Π)
π = eigen(Π′,sortby = x -> -abs(x)).vectors[:,1] # We know that 1 will be the largest eigenvalue
π = π/sum(π) # Normalize the eigenvector so it corresponds to a prob. distribution


#Pick an unstable initial distribution
π_0 = ones(3)/3
π_100 = transpose(π_0)*Π^100

# Check whether we have converged close enough to the invariant distribution π
error = norm(π_100 - transpose(π))
if error < 1e-8
    print("Converged")
else
    @show "Probability is $π_s_100" 
end