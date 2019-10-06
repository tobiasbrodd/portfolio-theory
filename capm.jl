module CAPM

# CAPM

include("utils.jl")

using Statistics, .Utils

export implied_returns

# Implied returns (CAPM)
function implied_returns(R, r, sigma, w)
    lambda = risk_aversion_coefficient(R, r)

    return lambda * sigma * w
end

end