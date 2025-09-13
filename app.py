def analyze_stock(ticker):
    hist, info = fetch_stock_data(ticker)
    
    if hist.empty:
        return {
            "Ticker": ticker,
            "Recommendation": "Data Unavailable",
            "Confidence (%)": 0,
            "Price Action": "No historical data found",
            "Sentiment": "No data available"
        }
    
    pa_arg, pa_score = price_action_agent(hist)
    sent_arg, sent_score = sentiment_agent(ticker)
    
    final_score = (pa_score * 0.6) + (sent_score * 0.4)
    recommendation = "Buy" if final_score > 0.6 else "Hold"
    
    return {
        "Ticker": ticker,
        "Recommendation": recommendation,
        "Confidence (%)": round(final_score * 100, 2),
        "Price Action": pa_arg,
        "Sentiment": sent_arg
    }
