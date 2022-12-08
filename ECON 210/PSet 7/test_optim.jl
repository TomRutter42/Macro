using NLopt
function objective(x, grad)
    if length(grad > 0) 
        grad[1] = 2 * x[1]
        grad[2] = 2 * x[2]
    end
    return(x[1]^2 + x[2]^2)
end

opt = Opt(:LD_MMA, 2)

opt.max_objective = objective
opt.lower_bounds = [0.001, 0]
opt.upper_bounds = [200, 1]
opt.xtol_abs = 10^(-30)

println("Guess Outcome: ", objective([200 / 2, 0.9]))
println("Guess Outcome: ", objective([200 / 4, 0.9]))

(maxf, maxpol, ret) = optimize(opt, [200 / 2, 0.9])
numevals = opt.numevals # the number of function evaluations
println("got $maxf at $maxpol[1] and $maxpol[2] after $numevals iterations (returned $ret)")