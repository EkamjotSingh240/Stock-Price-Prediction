# def detect_trend(df):
#     """
#     Detect stock trend using moving averages
#     """

#     latest = df.iloc[-1]

#     ma10 = latest["MA10"]
#     ma50 = latest["MA50"]

#     if ma10 > ma50:
#         return "Bullish"

#     elif ma10 < ma50:
#         return "Bearish"

#     else:
#         return "Sideways"


def detect_trend(df):
    """
    Detect stock trend using moving averages over recent days
    """

    recent = df.tail(5)

    ma10_avg = recent["MA10"].mean()
    ma50_avg = recent["MA50"].mean()

    if ma10_avg > ma50_avg:
        return "Bullish"

    elif ma10_avg < ma50_avg:
        return "Bearish"

    else:
        return "Sideways"