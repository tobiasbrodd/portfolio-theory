module Backtester

include("bl.jl")
include("rb.jl")
include("mpt.jl")
include("capm.jl")
include("utils.jl")

using DataFrames, Dates, Printf, JSON, LinearAlgebra, Missings, Statistics, Cairo, Gadfly, .CAPM, .BL, .RB, .MPT, .Utils

function black_litterman_strategy(tickers, assets, market, market_weights, initial_capital, initial_date; risk_free_rate = 0.05, tau = 0.05, long = false, normalize = false, verbose = true)
    shares = get_initial_shares(tickers, assets, initial_date)
    dates = get_dates(initial_date, assets)

    for date in dates
        println(date)

        w_r, w = calculate_bl_weights(tickers, assets, market, market_weights, date, risk_free_rate = risk_free_rate, tau = tau, long = long)

        if normalize
            d = maximum(abs.([w_r; w]))
            w_r /= d
            w /= d
        end

        if verbose
            print_weights(tickers, w_r, w)
        end

        prices = assets[assets.Date .>= date, :]
        shares[shares.Date .>= date, :] .= calculate_shares(tickers, prices, initial_capital, w_r, w)
    end

    prices = assets[assets.Date .>= initial_date, :]
    signals = calculate_signals(tickers, shares)
    
    return prices, shares, signals
end

function risk_budgeting_strategy(tickers, assets, initial_capital, initial_date; verbose = true)
    shares = get_initial_shares(tickers, assets, initial_date)
    dates = get_dates(initial_date, assets)

    for date in dates
        println(date)

        w_r, w = calculate_rb_weights(tickers, assets, date)

        if verbose
            print_weights(tickers, w_r, w)
        end

        prices = assets[assets.Date .>= date, :]
        shares[shares.Date .>= date, :] .= calculate_shares(tickers, prices, initial_capital, w_r, w)
    end

    prices = assets[assets.Date .>= initial_date, :]
    signals = calculate_signals(tickers, shares)
    
    return prices, shares, signals
end

function calculate_bl_weights(tickers, assets, market, market_weights, date; risk_free_rate = 0.05, tau = 0.05, long = false)
    _, _, _, historical_returns = get_assets_data(tickers, assets, date)
    _, _, w_market, market_returns = get_market_data(market, market_weights, date)

    P, Q = get_views(tickers, historical_returns)

    historical_returns_matrix = convert(Matrix, historical_returns[:, 2:end])
    
    return black_litterman(historical_returns_matrix, market_returns, risk_free_rate, P, Q, tau, w_market, long = long)
end

function calculate_rb_weights(tickers, assets, date)
    _, _, _, historical_returns = get_assets_data(tickers, assets, date)

    historical_returns_matrix = convert(Matrix, historical_returns[:, 2:end])

    n = size(historical_returns_matrix, 2)
    budget = (1/n) * ones(n)
    
    return 0, cvar_budgeting(historical_returns_matrix, budget)
end

function get_views(tickers, historical_returns; days = 30)
    historical_returns_matrix = convert(Matrix, historical_returns[:, 2:end])

    P = Matrix(I, length(tickers), length(tickers))
    # Q = transpose(convert(Matrix, historical_returns_matrix[end:end,:])) .- 1
    Q = transpose(exp.(mean(log.(historical_returns_matrix .+ 1), dims = 1)) .^ days) .- 1

    println("View: ", round(mean(Q) * 100, digits=2), "%")

    return P, Q
end

function print_weights(tickers, w_r, w)
    println("----------")

    w_r = round(w_r * 100, digits=1)
    w = map(x -> round((x * 100), digits=1), w)
    println("Risk-free weight: " * string(w_r) * "%")
    for (i, ticker) in enumerate(tickers)
        println(ticker * ": " * string(w[i]) * "%")
    end

    println("----------")
end

function get_dates(initial_date, assets)
    dates = [initial_date]
    end_date = dates[end]
    
    while end_date < (assets.Date[end] - Dates.Month(1))
        end_date = dates[end] + Dates.Month(1)
        push!(dates, end_date)
    end

    return dates
end

function get_initial_date(first_date, assets)
    assets_date = minimum(assets.Date) + Dates.Month(1)
    market_date = minimum(assets.Date) + Dates.Month(1)
    
    return max(first_date, assets_date, market_date)
end

function get_initial_shares(tickers, assets, initial_date)
    shares = DataFrame(Date = assets[assets.Date .>= initial_date, :Date])
    
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

function get_assets_data(tickers, assets, date)
    # Get historical prices
    historical_prices = assets[assets.Date .< date, :]
    historical_returns = calculate_daily_ticker_returns(historical_prices, tickers)

    # Get future prices
    future_prices = assets[assets.Date .>= date, :]
    future_returns = calculate_daily_ticker_returns(future_prices, tickers)

    return future_prices, future_returns, historical_prices, historical_returns
end

function get_market_data(market, market_weights, date)
    # Get historical market
    historical_market = market[market.Date .< date, :]
    historical_market[!, :Returns] = (calculate_returns(historical_market, :Price) .- 1) * 100

    # Get future market
    future_market = market[market.Date .>= date, :]
    future_market[!, :Returns] = (calculate_returns(future_market, :Price) .- 1) * 100

    # Get market returns
    market_returns = calculate_daily_returns(historical_market, :Price) .- 1

    # Get market weights
    w_market = market_weights.Weight

    return future_market, historical_market, w_market, market_returns
end

function get_tickers()
    tickers = open("tickers.json", "r") do f
        global tickers
        txt = read(f, String)
        tickers = JSON.parse(txt)
    end

    return tickers
end

function get_usable_tickers(tickers, date; verbose = true)
    path = "csv/"
    ticker = tickers[1]
    df = get_dataframe(path = path, filename = ticker * ".csv", name = ticker)

    for i in 2:length(tickers)
        ticker = tickers[i]
        filename = ticker * ".csv"

        ticker_df = get_dataframe(path = path, filename = filename, name = ticker)
        df = join(df, ticker_df, on = :Date, kind = :outer)
    end

    usable_tickers = []
    for ticker in tickers
        sym = Symbol(ticker)
        first_date = minimum(dropmissing(df[:, [:Date, sym]]).Date)
        if verbose
            println(((first_date > date) ? "* " : "") * ticker * ": " * Dates.format(first_date, "yyyy-mm-dd"))
        end

        if first_date <= date
            push!(usable_tickers, ticker)
        end
    end

    return usable_tickers
end

function print_performance(portfolio)
    @printf "Days: %d\n" size(portfolio, 1)
    @printf "Return: %.2f%%\n" portfolio.Returns[end]
    @printf "Average daily return: %.2f%%\n" (exp.(mean(log.(portfolio.DailyReturns / 100 .+ 1))) .- 1) * 100
    @printf "Average monthly return: %.2f%%\n" (exp.(mean(log.(portfolio.DailyReturns / 100 .+ 1))) .^ 30 .- 1) * 100
end

function plot_perfomance(portfolio, market)
    strategy_layer = layer(portfolio, x=:Date, y=:Returns, Geom.line, Theme(default_color="blue"))
    market_layer = layer(market, x=:Date, y=:Returns, Geom.line, Theme(default_color="black"))
    
    theme = Theme(background_color=color("white"))

    strategy_plot = plot(strategy_layer, theme, Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Black-Litterman"), Guide.manual_color_key("", ["Portfolio"], ["blue"]))
    strategy_plot |> PNG("plots/black_litterman.png", 8inch, 5inch)

    market_plot = plot(market_layer, theme, Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Market"), Guide.manual_color_key("", ["Market"], ["black"]))
    market_plot |> PNG("plots/market.png", 8inch, 5inch)

    comparison_plot = plot(strategy_layer, market_layer, theme, Guide.xlabel("Time"), Guide.ylabel("Return (%)"), Guide.title("Comparison"), Guide.manual_color_key("", ["Portfolio", "Market"], ["blue", "black"]))
    comparison_plot |> PNG("plots/comparison.png", 8inch, 5inch)
end

function run(tickers, first_date; long = false, normalize = false, verbose = true)
    # Get asset prices
    assets = get_prices(tickers)

    # Get market weights
    # w_market = ones(length(tickers),) ./ length(tickers)

    # Get market
    # market = DataFrame(Date = assets.Date)
    # price_matrix = convert(Matrix, assets[:, 2:end])
    # market[!, :Price] = price_matrix * w_market

    # Get market
    index_ticker = "OMX"
    market, market_weights = get_market(index_ticker)
    market_weights = market_weights[map(t -> t in tickers, market_weights.Ticker), :]

    initial_capital = 1000
    initial_date = get_initial_date(first_date, assets)

    # prices, shares, signals = black_litterman_strategy(tickers, assets, market, market_weights, initial_capital, initial_date; long = long, normalize = normalize, verbose = verbose)
    prices, shares, signals = risk_budgeting_strategy(tickers, assets, initial_capital, initial_date; verbose = verbose)
    portfolio = get_portfolio(tickers, prices, shares, signals, initial_capital)

    future_market, _, _, _ = get_market_data(market, market_weights, initial_date)
    print_performance(portfolio)
    # plot_perfomance(portfolio, future_market)
end

long = true
normalize = false

verbose = false
first_date = Dates.Date(2019, 1, 1)
tickers = get_tickers()
tickers = get_usable_tickers(tickers, first_date, verbose = verbose)
run(tickers, first_date, long = long, normalize = normalize, verbose = verbose)

end