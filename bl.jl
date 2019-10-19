module BL

# Black-Litterman Model

include("mpt.jl")
include("utils.jl")
include("capm.jl")

using Statistics, .MPT, .CAPM, .Utils

export allocate

# Black-Litterman
function black_litterman(P, Q, sigma, theta, tau)
    sigma_mu_inv = inv(tau * sigma)
    P_transpose = transpose(P)
    c = 1 / tau
    omega = (1 / c) * P * sigma * P_transpose

    mu_bl = inv(sigma_mu_inv + P_transpose * omega * P) * (sigma_mu_inv * theta + P_transpose * Q)
    sigma_bl = (1 + tau) * sigma - tau^2 * sigma * P_transpose * inv(tau * P * sigma * P_transpose + omega) * P * sigma

    return mu_bl, sigma_bl
end

# Allocate portfolio 
function allocate(R_historic, R_market, risk_free_rate, P, Q, tau, w)
    lambda = risk_aversion_coefficient(R_market, risk_free_rate)
    sigma = cov(R_historic)
    theta = implied_returns(R_market, risk_free_rate, sigma, w)
    mu_bl, sigma_bl = black_litterman(P, Q, sigma, theta, tau)
    
    return mean_variance(1, abs(lambda), mu_bl, sigma_bl; r = risk_free_rate)
    # return mean_variance_long(1, abs(lambda), mu_bl, sigma_bl; r = risk_free_rate)
end

function test()
    R_historic = [1.0 1.0; 1.3 1.2; 1.5 1.1]
    R_market = [1.0 1.2 1.3]
    risk_free_rate = 1.05
    P = [1 0; 0 1]
    Q = [1.5; 1.0]
    tau = 0.05
    w = [0.5; 0.5]

    return allocate(R_historic, R_market, risk_free_rate, P, Q, tau, w)
end

end