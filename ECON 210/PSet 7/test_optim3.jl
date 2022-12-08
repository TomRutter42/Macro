using Optim

println("First Part")

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

x0 = [0.0, 0.0]

outcome1 = optimize(f, x0)

println("Second Part")

lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0] 

results = optimize(f, lower, upper, initial_x)