module Utils

# Utility Functions

using Statistics

export risk_aversion_coefficient, sample_sigma

# Risk Averision Coefficient
function risk_aversion_coefficient(R, r)
    return (mean(R) - r) / var(R)
end

# Sample Sigma
function sample_sigma(R)
    T, _ = size(R)
    R_mean = mean(R, dims=1)

    return (1 / T) * transpose(R) * R - transpose(R_mean) * R_mean
end

end