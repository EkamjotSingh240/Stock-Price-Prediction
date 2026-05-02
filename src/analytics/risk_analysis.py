# def calculate_risk(df):
#     """
#     Calculate risk level using volatility
#     """

#     latest = df.iloc[-1]

#     volatility = latest["Volatility"]

#     if volatility < 0.01:
#         return "Low Risk"

#     elif volatility < 0.03:
#         return "Medium Risk"

#     else:
#         return "High Risk"

def calculate_risk(df):
    """
    Calculate risk level using average volatility
    """

    volatility = df["Volatility"].tail(10).mean()

    if volatility < 0.01:
        return "Low Risk"

    elif volatility < 0.03:
        return "Medium Risk"

    else:
        return "High Risk"