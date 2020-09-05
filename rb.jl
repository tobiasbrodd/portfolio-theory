module RB

# Modern Portfolio Theory

using JuMP, Ipopt, Statistics, ForwardDiff

export std_budgeting, cvar_budgeting

# Risk Budgeting using Standard Deviation (Volatility) as a Risk Measure
function std_budgeting(R, rb)
    n = size(rb, 1)
    sigma = cov(R)
    std_rb(w...) = sum((w[i] * (sum(sigma[k,:] * w[k] for k=1:n)[i] / sum(w[l] * sum(sigma[k,l] * w[k] for k=1:n) for l=1:n)) - rb[i])^2 for i=1:n)

    return risk_budgeting(std_rb, rb)
end

# Risk Budgeting using Conditional Value at Risk (Expected Shortfall) as a Risk Measure
function cvar_budgeting(R, rb; p = 0.05)
    n = size(rb, 1)
    m = Int(floor(p * size(R, 1)))

    L = sort(-R, dims=1)

    # cvar(w...) = mean(sum(w[i] * L[k, i] for i = 1:n) for k = 1:m)
    cvar_rb(w...) = sum(w[i] * mean(L[k, i] for k = 1:m) - rb[i] * mean(sum(w[j] * L[k, j] for j = 1:n) for k = 1:m) for i = 1:n)

    return risk_budgeting(cvar_rb, rb)
end

# Risk Budgeting
function risk_budgeting(f, rb)
    n = size(rb, 1)
    model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
    set_silent(model)

    @variable(model, 0 <= w[1:n] <= 1)
    register(model, :f, n, f; autodiff=true)

    @NLobjective(model, Min, f(w...))
    @constraint(model, sum(w[i] for i=1:n) == 1)
    optimize!(model)

    return value.(w)
end

# Risk Budgeting
# function risk_budgeting_test(risk, rb)
#     n = size(rb, 1)

#     f = w -> risk(w...)
#     g(w) = ForwardDiff.gradient(f, w)
#     c_risk(w...) = g([w[i] for i=1:n])

#     println(c_risk(ones(n)...))

#     model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
#     set_silent(model)

#     @variable(model, 0 <= w[1:n] <= 1)
#     register(model, :risk, n, risk; autodiff=true)
#     register(model, :c_risk, n, c_risk; autodiff=true)

#     @NLobjective(model, Min, (1 / n) * sum(w[i] * c_risk(w...)[i] for i=1:n) - sum(rb[i] * risk(w...) for i=1:n))
#     @constraint(model, sum(w[i] for i=1:n) == 1)
#     optimize!(model)

#     return value.(w)
# end

end