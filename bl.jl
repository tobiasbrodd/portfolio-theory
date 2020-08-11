module BL

# Black-Litterman Model

include("mpt.jl")
include("utils.jl")
include("capm.jl")

using Statistics, LinearAlgebra, .MPT, .CAPM, .Utils

export allocate

# Black-Litterman
function black_litterman(P, Q, sigma, theta, tau)
    tau_sigma_inv = inv(tau * sigma)
    P_transpose = transpose(P)
    c = 1 / tau
    omega = (1 / c) * Diagonal(P * sigma * P_transpose)
    omega_inv = inv(omega)

    mu_bl = inv(tau_sigma_inv + P_transpose * omega_inv * P) * (tau_sigma_inv * theta + P_transpose * omega_inv * Q)
    sigma_bl = sigma + inv(tau_sigma_inv + P_transpose * omega_inv * P)

    # mu_bl = theta + tau * sigma * P_transpose * inv(P * tau * sigma * P_transpose + omega) * (Q - P * theta)
    # sigma_bl = (1 + tau) * sigma - tau^2 * sigma * P_transpose * inv(tau * P * sigma * P_transpose + omega) * P * sigma

    return mu_bl, sigma_bl
end

# Allocate portfolio 
function allocate(R_historic, R_market, risk_free_rate, P, Q, tau, w; long = false)
    lambda = risk_aversion_coefficient(R_market, risk_free_rate)
    sigma = cov(R_historic)
    theta = implied_returns(R_market, risk_free_rate, sigma, w)
    mu_bl, sigma_bl = black_litterman(P, Q, sigma, theta, tau)

    println("CAPM: ", round(mean(theta) * 100, digits=2), "%")
    println("Black-Litterman: ", round(mean(mu_bl) * 100, digits=2), "%")
    
    if long
        return mean_variance_long(1, abs(lambda), mu_bl, sigma_bl; r = risk_free_rate)
    else
        return mean_variance(1, abs(lambda), mu_bl, sigma_bl; r = risk_free_rate)
    end
end

function test()
    R_historic = [0.5 -0.3; 0.6 -0.2; 0.7 0.1]
    R_market = [0.1 0.2 0.3]
    risk_free_rate = 0.05
    P = [1 0; 0 1]
    Q = [0.8; 0.4]
    tau = 0.05
    w = [0.5; 0.5]

    return allocate(R_historic, R_market, risk_free_rate, P, Q, tau, w)
end

end