# def generate_signal(df):
#     """
#     Generate Buy/Sell/Hold signal using RSI
#     """

#     latest = df.iloc[-1]

#     rsi = latest["RSI"]

#     if rsi < 30:
#         return "BUY"

#     elif rsi > 70:
#         return "SELL"

#     else:
#         return "HOLD"

def generate_signal(df):
    """
    Generate buy/sell signal using RSI and MACD
    """

    latest = df.iloc[-1]

    rsi = latest["RSI"]
    macd = latest["MACD"]

    if rsi < 30 and macd > 0:
        return "BUY"

    elif rsi > 70 and macd < 0:
        return "SELL"

    else:
        return "HOLD"