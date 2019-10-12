module MPT

# Modern Portfolio Theory

using JuMP, Ipopt

export mean_variance, mean_variance_long, expectation_maximization, variance_minimization, minimum_variance

# Mean Variance Portfolio (Trade-off Problem)
function mean_variance(v, lambda, mu, sigma; r = nothing)
    n = size(sigma)[1]
    o = ones(n, 1)
    sigma_inv = inv(sigma)

    if isnothing(r)
        multiplier = (transpose(mu) * sigma_inv * o - lambda) / (transpose(o) * sigma_inv * o)
        multiplier = max(multiplier, 0)

        w = (v / lambda) * sigma_inv * (mu - multiplier * o)

        return 0, w
    else
        w = (v / lambda) * sigma_inv * (mu - r * o)
        w_r = v - (transpose(w) * o)[1]

        return w_r, w
    end
end


# Mean Variance Portfolio (Trade-off Problem) - Long Positions Only
function mean_variance_long(v, lambda, mu, sigma; r = nothing)
    n = size(mu, 1)
    m = Model(with_optimizer(Ipopt.Optimizer))

    @variable(m, 0 <= w[1:n] <= 1)
    @NLobjective(m, Max, sum(w[i] * mu[i] for i=1:n) + sum(w[j] * sum(sigma[i,j] * w[i] for i=1:n) for j=1:n))

    if isnothing(r)
        @constraint(m, sum(w[i] for i=1:n) == v)
        optimize!(m)

        return value.(w)
    else
        @variable(m, 0 <= w_r <= 1)
        @constraint(m, w_r + sum(w[i] for i=1:n) == v)
        optimize!(m)

        return value(w_r), value.(w)
    end
end

# Expectation Maximazation Portfolio
function expectation_maximization(v, s, r, mu, sigma)
    n = size(sigma)[1]
    o = ones(n, 1)
    sigma_inv = inv(sigma)
    mu_r = mu - r * o

    w = s * v * (sigma_inv * mu_r) / sqrt(transpose(mu_r) * sigma_inv * mu_r)
    w_r = v - (transpose(w) * o)[1]

    return w_r, w
end

# Variance Minimization Portfolio
function variance_minimization(v, m, r, mu, sigma)
    n = size(sigma)[1]
    o = ones(n, 1)
    sigma_inv = inv(sigma)
    mu_r = mu - r * o

    w = v * (m - r) * (sigma_inv * mu_r) / (transpose(mu_r) * sigma_inv * mu_r)
    w_r = v - (transpose(w) * o)[1]

    return w_r, w
end

# Minimum Variance Portfolio
function minimum_variance(v, mu, sigma)
    n = size(sigma)[1]
    o = ones(n, 1)
    sigma_inv = inv(sigma)

    w = v * (sigma_inv * o) / (transpose(o) * sigma_inv * o)

    return 0, w
end

end