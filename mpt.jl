module MPT

# Modern Portfolio Theory
export trade_off_problem, expectation_maximization, variance_minimization, minimum_variance

# Trade-off Problem Portfolio
function trade_off_problem(v, c, mu, sigma; r = nothing)
    n = size(sigma)[1]
    o = ones(n, 1)
    sigma_inv = inv(sigma)

    if isnothing(r)
        lambda = (transpose(mu) * sigma_inv * o - c) / (transpose(o) * sigma_inv * o)
        lambda = max(lambda, 0)

        w = (v / c) * sigma_inv * (mu - lambda * o)

        return 0, w
    else
        w = (v / c) * sigma_inv * (mu - r * o)
        w_r = v - (transpose(w) * o)[1]

        return w_r, w
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