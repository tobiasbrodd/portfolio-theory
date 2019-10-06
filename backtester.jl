module Backtester

include("bl.jl")
include("capm.jl")
include("utils.jl")

using DataFrames, Dates, LinearAlgebra, Statistics, Gadfly, .CAPM, .BL, .Utils

function black_litterman_strategy(prices, tickers, historic_returns, market_returns, risk_free_rate, P, Q, tau, w_market, initial_capital)
    w_r, w = allocate(historic_returns, market_returns, risk_free_rate, P, Q, tau, w_market)

    w_r /= 10000
    w /= 10000

    println(w_r, w)

    shares = deepcopy(prices)
    for i in 1:length(tickers)
        sym = Symbol(tickers[i])
        c = initial_capital * w[i]
        s = floor(c / prices[1, sym])

        shares[!,sym] = ones(nrow(shares),) * s
    end

    c = initial_capital * w_r
    s = floor(c / prices[1, :Bond])
    shares[!,:Bond] = ones(nrow(shares),) * s

    signals = deepcopy(shares)
    for i in 1:length(tickers)
        sym = Symbol(tickers[i])
        signals[2:end, sym] = diff(signals[:, sym])
    end

    signals[2:end, :Bond] = diff(signals[:, :Bond])

    return shares, signals
end

function portfolio(prices, shares, signals, tickers, initial_capital)
    p = DataFrame(Date = prices.Date)

    p[!, :Holdings] = zeros(nrow(p),)
    for i in 1:length(tickers)
        sym = Symbol(tickers[i])
        p.Holdings += shares[:, sym] .* prices[:, sym]
    end

    p.Holdings += shares[:, :Bond] .* prices[:, :Bond]

    p[!, :Cash] = ones(nrow(p),) * initial_capital
    for i in 1:length(tickers)
        sym = Symbol(tickers[i])
        p.Cash -= cumsum(signals[:, sym] .* prices[:, sym])
    end

    p.Cash -= cumsum(signals[:, :Bond] .* prices[:, :Bond])

    p[!, :Total] = p.Holdings .+ p.Cash
    p[!, :Returns] = (cumprod([1.0; p.Total[2:end] ./ p.Total[1:end-1]]) .- 1) * 100

    return p
end

function plot_perfomance(p, index)
    strategy = layer(p, x=:Date, y=:Returns, Geom.line, Theme(default_color="blue"))

    strategy_p = plot(strategy, Theme(key_position=:none), Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Black-Litterman"))
    strategy_p |> SVG("plots/black_litterman.svg", 15inch, 8inch)

    market = layer(index, x=:Date, y=:Returns, Geom.line, Theme(default_color="black"))

    market_p = plot(market, Theme(key_position=:none), Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Market"))
    market_p |> SVG("plots/market.svg", 15inch, 8inch)

    comparison_p = plot(strategy, market, Theme(key_position=:none), Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Comparison"))
    comparison_p |> SVG("plots/comparison.svg", 15inch, 8inch)
end

tickers = ["HM_B", "NDA_SE", "TELIA"]
index_ticker = "OMXS30"

w_market = ones(length(tickers),) ./ length(tickers)

prices = get_prices(tickers)
historic_prices = prices[1:1100, :]
historic_returns = calculate_returns(historic_prices, tickers)
historic_returns_matrix = convert(Matrix, historic_returns[:, 2:end])
prices = prices[1101:end, :]
returns = calculate_returns(prices, tickers)
returns_matrix = convert(Matrix, returns[:, 2:end])

historic_index = DataFrame(Date = historic_prices.Date)
historic_index[!, :DailyReturns] = historic_returns_matrix * w_market
historic_index[!, :Returns] = (cumprod(historic_index.DailyReturns) .- 1) * 100
index = DataFrame(Date = prices.Date)
index[!, :DailyReturns] = returns_matrix * w_market
index[!, :Returns] = (cumprod(index.DailyReturns) .- 1) * 100

market_returns = historic_index.DailyReturns
risk_free_rate = 1.0
tau = 0.05

P = Matrix(I, length(tickers), length(tickers))
Q = transpose(convert(Matrix, historic_returns_matrix[end:end,:]))

prices[!, :Bond] = ones(nrow(prices),)
returns[!, :Bond] = ones(nrow(returns),) * risk_free_rate

initial_capital = 1000

shares, signals = black_litterman_strategy(prices, tickers, historic_returns_matrix, market_returns, risk_free_rate, P, Q, tau, w_market, initial_capital)
p = portfolio(prices, shares, signals, tickers, initial_capital)

plot_perfomance(p, index)
println(p.Returns[end])

end