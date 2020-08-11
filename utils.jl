module Utils

# Utility Functions

using DataFrames, CSV, DelimitedFiles, Statistics, Missings

export risk_aversion_coefficient, sample_sigma, get_prices, get_market, calculate_daily_ticker_returns, calculate_daily_returns, calculate_returns, get_dataframe, format_stock_csv, format_index_csv

# Risk averision coefficient
function risk_aversion_coefficient(R, r)
    return (mean(R) - r) / var(R)
end

# Sample sigma
function sample_sigma(R)
    T, _ = size(R)
    R_mean = mean(R, dims = 1)

    return (1 / T) * transpose(R) * R - transpose(R_mean) * R_mean
end

# Get prices
function get_prices(tickers)
    path = "csv/"
    ticker = tickers[1]
    df = get_dataframe(path = path, filename = ticker * ".csv", name = ticker)

    for i in 2:length(tickers)
        ticker = tickers[i]
        filename = ticker * ".csv"

        ticker_df = get_dataframe(path = path, filename = filename, name = ticker)
        df = join(df, ticker_df, on = :Date, kind = :outer)
    end

    dropmissing!(df)
end

# Get market
function get_market(index_ticker)
    path = "csv/"

    df_index = get_dataframe(path = path, filename = index_ticker * ".csv", name = "Price")
    df_market = CSV.File(path * "market.csv") |> DataFrame
    rename!(df_market, :Column1 => :Ticker)
    rename!(df_market, :weight => :Weight)
    df_market = df_market[:, [:Ticker, :Weight]]

    return df_index, df_market
end

# Get returns
function calculate_daily_ticker_returns(prices, tickers)
    returns = DataFrame(Date = prices.Date)

    for ticker in tickers
        col = Symbol(ticker)
        returns[!, col] = calculate_daily_returns(prices, col) .- 1
    end

    return returns
end

# Calculate daily returns
function calculate_daily_returns(prices, col)
    return [1.0; prices[2:end,col] ./ prices[1:end-1,col]] 
end

# Calculate returns (cumulative)
function calculate_returns(prices, col)
    daily_returns = calculate_daily_returns(prices, col)

    return cumprod(daily_returns)
end

# Get CSV dataframe
function get_dataframe(;path, filename, name)
    df = CSV.File(path * filename) |> DataFrame

    return format_dataframe(df, name = name)
end

# Format dataframe
function format_dataframe(df; name)
    df = df[:, [:Date, :Close]]
    df = coalesce.(df, 0)
    df = df[(df.Close .!= 0), :]
    rename!(df, :Close => Symbol(name))
    sort!(df, :Date)

    return df
end

# Format CSV file
function format_stock_csv(;path, filename)
    lines = String[]

    open(path * filename) do file
        for line in eachline(file)
            line = replace(line, "," => ".")
            line = replace(line, ";" => ",")
            line = line[1:end-1]
            push!(lines, line)
        end
    end

    open(path * filename, "w") do file
        writedlm(file, lines)
    end
end

# Format CSV file
function format_index_csv(;path, filename)
    lines = String[]

    open(path * filename) do file
        for line in eachline(file)
            line = replace(line, "," => "")
            line = replace(line, ";" => ",")
            line = line[1:end-1]
            push!(lines, line)
        end
    end

    open(path * filename, "w") do file
        writedlm(file, lines)
    end
end

end