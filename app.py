import yfinance as yf
import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
import requests

# --- Caching pipelines for faster execution ---
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_sentiment_pipeline()

@st.cache_data
def fetch_stock_data(ticker, period="6mo"):
    data = yf.Ticker(ticker)
    hist = data.history(period=period)
    info = data.info
    return hist, info

# --- Phase 5: Dynamic top stocks selection ---
@st.cache_data
def get_top_stocks(n=10):
    """
    Fetch top active NSE stocks by volume.
    Returns a list of tickers with .NS suffix.
    """
    try:
        url = "https://www.nseindia.com/api/live-analysis-variations?index=eq"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.5"
        }
        session = requests.Session()
        # Initial request to get cookies
        session.get("https://www.nseindia.com", headers=headers)
        resp = session.get(url, headers=headers).json()
        # Extract data (adjust keys if NSE changes)
        df = pd.DataFrame(resp['data'])
        df = df.sort_values('tradedQuantity', ascending=False)
        tickers = df['symbol'].head(n).tolist()
        tickers = [t + ".NS" for t in tickers]
        return tickers
    except Exception as e:
        st.warning(f"Error fetching NSE top stocks: {e}")
        return ['VBL.NS','DABUR.NS','PFC.NS','ABDL.NS','JIO.NS','FINANCE.NS'][:n]

# --- Agents ---
def price_action_agent(hist):
    ma10 = hist['Close'].rolling(10).mean().iloc[-1]
    ma30 = hist['Close'].rolling(30).mean().iloc[-1]
    last = hist['Close'].iloc[-1]
    if ma10 > ma30:
        return "Bullish MA", 0.4, last*1.01, last*1.05
    else:
        return "Bearish MA", 0.2, last*0.99, last*0.95

def technical_agent(hist):
    close = hist['Close']
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2*std20
    last_close = close.iloc[-1]
    bullish = last_close < upper.iloc[-1] and rsi.iloc[-1] < 70
    if bullish:
        return f"Technical Bullish (RSI={round(rsi.iloc[-1],1)})", 0.3, last_close*1.01, last_close*1.05
    else:
        return f"Technical Bearish/neutral (RSI={round(rsi.iloc[-1],1)})", 0.15, last_close*0.99, last_close*0.95

def sentiment_agent(ticker):
    headlines = [
        f"{ticker} reports strong earnings",
        f"{ticker} faces regulatory risks"
    ]
    sentiments = [sentiment_analyzer(h)[0] for h in headlines]
    pos = sum(1 for s in sentiments if s['label']=='POSITIVE')
    score = 0.2 if pos >=1 else 0.1
    return f"{pos}/2 positive", score, None, None

def fundamental_agent(info):
    pe = info.get('trailingPE', None)
    roe = info.get('returnOnEquity', None)
    if pe and pe<25 and roe and roe>0.15:
        return f"Strong fundamentals (PE={pe}, ROE={round(roe,2)})", 0.3, None, None
    else:
        return f"Moderate fundamentals (PE={pe}, ROE={roe})", 0.15, None, None

def volume_agent(hist):
    v10 = hist['Volume'].rolling(10).mean().iloc[-1]
    v30 = hist['Volume'].rolling(30).mean().iloc[-1]
    if v10>1.5*v30:
        return "Volume spike", 0.1, None, None
    else:
        return "Volume normal", 0.05, None, None

# --- Multi-round debate moderator ---
def moderator(debate_dict):
    scores = [score for _,score,_,_ in debate_dict.values()]
    disagreement = max(scores)-min(scores)
    adjusted = {}
    for agent,(arg,score,entry,exit_level) in debate_dict.items():
        if disagreement>0.3 and score==max(scores):
            adjusted[agent]=(arg+" (boost)", min(score+0.05,1.0), entry, exit_level)
        else:
            adjusted[agent]=(arg,score,entry,exit_level)
    return adjusted

# --- Analyze single stock ---
def analyze_stock(ticker):
    hist, info = fetch_stock_data(ticker)
    if hist.empty:
        return {"Ticker": ticker, "Recommendation":"Data Unavailable","Confidence (%)":0}
    debate = {}
    debate['Price Action'] = price_action_agent(hist)
    debate['Technical'] = technical_agent(hist)
    debate['Sentiment'] = sentiment_agent(ticker)
    debate['Fundamentals'] = fundamental_agent(info)
    debate['Volume'] = volume_agent(hist)
    adjusted = moderator(debate)
    final_score = sum(score for _,score,_,_ in adjusted.values())
    recommendation = "Buy" if final_score>=0.6 else "Hold"
    risk_level = "High-risk" if final_score>=0.7 else "Low-risk"
    transcript = {agent:arg for agent,(arg,_,_,_) in adjusted.items()}
    entry_exit = {agent:(entry,exit_level) for agent,(_,_,entry,exit_level) in adjusted.items() if entry and exit_level}
    return {"Ticker":ticker,"Recommendation":recommendation,"Confidence (%)":round(final_score*100,2),
            "Risk Level":risk_level,"Debate Transcript":transcript,"Entry/Exit Levels":entry_exit}

# --- Streamlit UI ---
st.title("Multi-Agent Stock Recommendation System (Phases 1-5)")

if st.button("Generate Recommendations"):
    tickers = get_top_stocks(n=10)
    results = [analyze_stock(t) for t in tickers]
    df = pd.DataFrame(results)
    df_expanded = df.join(pd.json_normalize(df.pop("Debate Transcript")))
    df_expanded = df_expanded.join(pd.json_normalize(df.pop("Entry/Exit Levels")))
    st.table(df_expanded)
    if st.button("Export CSV"):
        df_expanded.to_csv("stock_recommendations.csv", index=False)
        st.success("Report saved")
