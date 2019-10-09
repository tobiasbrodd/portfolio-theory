module Backtester

include("bl.jl")
include("capm.jl")
include("utils.jl")

using DataFrames, Dates, Printf, LinearAlgebra, Statistics, Gadfly, .CAPM, .BL, .Utils

function black_litterman_strategy(tickers, assets, market, initial_capital, first_date)
    shares = get_initial_shares(tickers, assets)
    dates = get_dates(first_date)

    for date in dates
        println(date)

        w_r, w = calculate_weights(tickers, assets, market, date)
        d = maximum(abs.([w_r; w]))
        w_r /= d
        w /= d

        println("Risk-free weight: ", w_r)
        println("Risky weights: ", w)

        prices = assets[assets.Date .>= date, :]
        shares[shares.Date .>= date, :] .= calculate_shares(tickers, prices, initial_capital, w_r, w)
    end

    prices = assets[assets.Date .>= first_date, :]
    signals = calculate_signals(tickers, shares)
    
    return prices, shares, signals
end

function get_dates(first_date)
    dates = [first_date]
    end_date = dates[end]
    
    while end_date < (assets.Date[end] - Dates.Month(1))
        end_date = dates[end] + Dates.Month(1)
        push!(dates, end_date)
    end

    return dates
end

function calculate_weights(tickers, assets, market, date)
    risk_free_rate = 1.0
    tau = 0.05

    future_prices, _, _, historical_returns = get_assets_data(tickers, assets, risk_free_rate, date)
    _, _, w_market, market_returns = get_market_data(tickers, market, date)

    P, Q = get_views(tickers, historical_returns)

    historical_returns_matrix = convert(Matrix, historical_returns[:, 2:end])
    
    return allocate(historical_returns_matrix, market_returns, risk_free_rate, P, Q, tau, w_market)
end

function get_initial_shares(tickers, assets)
    shares = DataFrame(Date = assets[assets.Date .>= first_date, :Date])
    
    for ticker in tickers
        sym = Symbol(ticker)
        shares[!, sym] = zeros(size(shares, 1),)
    end

    shares[!, :Bond] = zeros(size(shares, 1),)

    return shares
end

function calculate_shares(tickers, prices, initial_capital, w_r, w)
    shares = DataFrame(Date = prices.Date)

    for i in 1:length(tickers)
        sym = Symbol(tickers[i])
        c = initial_capital * w[i]
        s = floor(c / prices[1, sym])

        shares[!, sym] = ones(nrow(shares),) * s
    end

    c = initial_capital * w_r
    shares[!,:Bond] = ones(nrow(shares),) * floor(c)

    return shares
end

function calculate_signals(tickers, shares)
    signals = deepcopy(shares)

    for ticker in tickers
        sym = Symbol(ticker)
        signals[2:end, sym] = diff(signals[:, sym])
    end

    signals[2:end, :Bond] = diff(signals[:, :Bond])

    return signals
end

function get_portfolio(tickers, prices, shares, signals, initial_capital)
    portfolio = DataFrame(Date = prices.Date)

    portfolio[!, :Holdings] = zeros(nrow(portfolio),)
    for ticker in tickers
        sym = Symbol(ticker)
        portfolio.Holdings += shares[:, sym] .* prices[:, sym]
    end

    portfolio.Holdings += shares[:, :Bond]

    portfolio[!, :Cash] = ones(nrow(portfolio),) * initial_capital
    for ticker in tickers
        sym = Symbol(ticker)
        portfolio.Cash -= cumsum(signals[:, sym] .* prices[:, sym])
    end

    portfolio.Cash -= cumsum(signals[:, :Bond])

    portfolio[!, :Total] = portfolio.Holdings .+ portfolio.Cash
    portfolio[!, :DailyReturns] = (calculate_daily_returns(portfolio, :Total) .- 1) * 100
    portfolio[!, :Returns] = (calculate_returns(portfolio, :Total) .- 1) * 100

    return portfolio
end

function print_performance(portfolio)
    @printf "Return: %.2f%%\n" portfolio.Returns[end]
    @printf "Average daily return: %.2f%%\n" mean(portfolio.DailyReturns)
end

function plot_perfomance(portfolio, market)
    strategy_layer = layer(portfolio, x=:Date, y=:Returns, Geom.line, Theme(default_color="blue"))
    market_layer = layer(market, x=:Date, y=:Returns, Geom.line, Theme(default_color="black"))

    strategy_plot = plot(strategy_layer, Theme(key_position=:none), Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Black-Litterman"))
    strategy_plot |> SVG("plots/black_litterman.svg", 15inch, 8inch)

    market_plot = plot(market_layer, Theme(key_position=:none), Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Market"))
    market_plot |> SVG("plots/market.svg", 15inch, 8inch)

    comparison_plot = plot(strategy_layer, market_layer, Theme(key_position=:none), Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Comparison"))
    comparison_plot |> SVG("plots/comparison.svg", 15inch, 8inch)
end

function get_assets_data(tickers, assets, risk_free_rate, date)
    # Get historical prices
    historical_prices = assets[assets.Date .< date, :]
    historical_returns = calculate_daily_ticker_returns(historical_prices, tickers)

    # Get future prices
    future_prices = assets[assets.Date .>= date, :]
    future_returns = calculate_daily_ticker_returns(future_prices, tickers)

    return future_prices, future_returns, historical_prices, historical_returns
end

function get_market_data(tickers, market, date)
    # Get market weights
    w_market = ones(length(tickers),) ./ length(tickers)

    # Get historical market
    historical_market = market[market.Date .< date, :]
    historical_market[!, :Returns] = (calculate_returns(historical_market, :Prices) .- 1) * 100

    # Get future market
    future_market = market[market.Date .>= date, :]
    future_market[!, :Returns] = (calculate_returns(future_market, :Prices) .- 1) * 100

    # Get market returns
    market_returns = calculate_daily_returns(historical_market, :Prices)

    return future_market, historical_market, w_market, market_returns
end

function get_views(tickers, historical_returns)
    historical_returns_matrix = convert(Matrix, historical_returns[:, 2:end])

    P = Matrix(I, length(tickers), length(tickers))
    Q = transpose(convert(Matrix, historical_returns_matrix[end:end,:]))

    return P, Q
end

# Get asset prices
tickers = ["HM_B", "NDA_SE", "TELIA"]
assets = get_prices(tickers)

# Get market weights
w_market = ones(length(tickers),) ./ length(tickers)

# Get market
market = DataFrame(Date = assets.Date)
price_matrix = convert(Matrix, assets[:, 2:end])
market[!, :Prices] = price_matrix * w_market

initial_capital = 1000
first_date = Dates.Date(2014, 1, 1)

prices, shares, signals = black_litterman_strategy(tickers, assets, market, initial_capital, first_date)
portfolio= get_portfolio(tickers, prices, shares, signals, initial_capital)

future_market, _, _, _ = get_market_data(tickers, market, first_date)
print_performance(portfolio)
plot_perfomance(portfolio, future_market)

end