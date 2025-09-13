import yfinance as yf
import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np

# --- Sentiment pipeline ---
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_sentiment_pipeline()

# --- Stock tickers pool ---
tickers_pool = [
    'RELIANCE.NS','TCS.NS','INFY.NS','HDFCBANK.NS','ICICIBANK.NS','KOTAKBANK.NS',
    'LT.NS','SBIN.NS','BAJFINANCE.NS','AXISBANK.NS','BHARTIARTL.NS','MARUTI.NS',
    'HINDUNILVR.NS','ITC.NS','TECHM.NS','WIPRO.NS','SUNPHARMA.NS','ADANIENT.NS',
    'JSWSTEEL.NS','ONGC.NS','TATAMOTORS.NS','ULTRACEMCO.NS','HCLTECH.NS','BPCL.NS',
    'EICHERMOT.NS','GRASIM.NS','VEDL.NS','TITAN.NS','INDUSINDBK.NS','NTPC.NS'
]

# --- Fetch stock data ---
@st.cache_data
def fetch_stock_data(ticker, period="6mo"):
    data = yf.Ticker(ticker)
    hist = data.history(period=period)
    info = data.info
    return hist, info

# --- Dynamic top stocks by average volume ---
def get_top_stocks_yf(tickers, n=5):
    stock_volumes = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="1mo")
            avg_volume = data['Volume'].mean()
            stock_volumes.append((ticker, avg_volume))
        except:
            continue
    stock_volumes.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in stock_volumes[:n]]

@st.cache_data
def get_top_stocks(n=5):
    return get_top_stocks_yf(tickers_pool, n)

# --- Multi-agent analysis ---
def price_action_agent(hist):
    ma10 = hist['Close'].rolling(10).mean().iloc[-1]
    ma30 = hist['Close'].rolling(30).mean().iloc[-1]
    last = hist['Close'].iloc[-1]
    if ma10 > ma30:
        return "Bullish MA", 0.3, last*1.01, last*1.05
    else:
        return "Bearish MA", 0.15, last*0.99, last*0.95

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
        return f"Technical Bullish (RSI={round(rsi.iloc[-1],1)})", 0.25, last_close*1.01, last_close*1.05
    else:
        return f"Technical Bearish/neutral (RSI={round(rsi.iloc[-1],1)})", 0.15, last_close*0.99, last_close*0.95

def sentiment_agent(ticker):
    headlines = [
        f"{ticker} reports strong earnings",
        f"{ticker} faces regulatory risks"
    ]
    sentiments = [sentiment_analyzer(h)[0] for h in headlines]
    pos = sum(1 for s in sentiments if s['label']=='POSITIVE')
    score = 0.15 if pos >=1 else 0.1
    return f"{pos}/2 positive", score, None, None

def fundamental_agent(info):
    pe = info.get('trailingPE', None)
    roe = info.get('returnOnEquity', None)
    if pe and pe<25 and roe and roe>0.15:
        return f"Strong fundamentals (PE={pe}, ROE={round(roe,2)})", 0.2, None, None
    else:
        return f"Moderate fundamentals (PE={pe}, ROE={roe})", 0.1, None, None

def volume_agent(hist):
    v10 = hist['Volume'].rolling(10).mean().iloc[-1]
    v30 = hist['Volume'].rolling(30).mean().iloc[-1]
    if v10>1.5*v30:
        return "Volume spike", 0.1, None, None
    else:
        return "Volume normal", 0.1, None, None

# --- Moderator ---
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

# --- Analyze stock ---
def analyze_stock(ticker):
    hist, info = fetch_stock_data(ticker)
    if hist.empty:
        return {"Ticker": ticker, "Recommendation":"Data Unavailable","Confidence (%)":0, 
                "Risk Level":"N/A", "Debate Transcript":{}, "Entry/Exit Levels":{}}

    debate = {}
    debate['Price Action'] = price_action_agent(hist)
    debate['Technical'] = technical_agent(hist)
    debate['Sentiment'] = sentiment_agent(ticker)
    debate['Fundamentals'] = fundamental_agent(info)
    debate['Volume'] = volume_agent(hist)

    adjusted = moderator(debate)

    # --- Normalize confidence ---
    agent_weights = [0.3,0.25,0.15,0.2,0.1]  # max possible for each agent
    max_possible_score = sum(agent_weights)
    final_score = sum(score for _,score,_,_ in adjusted.values())
    final_score_normalized = min(final_score / max_possible_score, 1.0)

    # --- Optional volatility adjustment ---
    volatility = hist['Close'].pct_change().rolling(30).std().iloc[-1]
    final_score_normalized *= max(0.8, 1 - volatility)  # reduces confidence if volatile

    confidence_percent = round(final_score_normalized*100,2)

    # --- Risk levels ---
    if final_score_normalized >= 0.7:
        risk_level = "High-risk"
    elif final_score_normalized >= 0.4:
        risk_level = "Medium-risk"
    else:
        risk_level = "Low-risk"

    transcript = {agent:arg for agent,(arg,_,_,_) in adjusted.items()}
    entry_exit = {agent:(entry,exit_level) for agent,(_,_,entry,exit_level) in adjusted.items() if entry and exit_level}

    return {"Ticker":ticker,
            "Recommendation":"Buy" if final_score_normalized>=0.6 else "Hold",
            "Confidence (%)":confidence_percent,
            "Risk Level":risk_level,
            "Debate Transcript":transcript,
            "Entry/Exit Levels":entry_exit}

# --- Streamlit UI ---
st.title("Multi-Agent Stock Recommendation System (Final Version)")

def format_entry_exit(d):
    if not isinstance(d, dict):
        return {}
    formatted = {}
    for k, v in d.items():
        if v and isinstance(v, tuple):
            formatted[k] = (round(float(v[0]),2), round(float(v[1]),2))
    return formatted

def format_debate(d):
    if not isinstance(d, dict):
        return "No debate data"
    formatted = [f"{k}: {v}" for k,v in d.items()]
    return "\n".join(formatted)

if st.button("Generate Recommendations"):
    tickers = get_top_stocks(n=5)
    results = [analyze_stock(t) for t in tickers]
    df = pd.DataFrame(results)

    df['Debate Transcript'] = df['Debate Transcript'].apply(format_debate)
    df['Entry/Exit Levels'] = df['Entry/Exit Levels'].apply(format_entry_exit)

    # Main table
    def color_risk(val):
        if val=="High-risk":
            return "background-color: #ff9999"
        elif val=="Medium-risk":
            return "background-color: #fff799"
        else:
            return "background-color: #b3ffb3"

    st.dataframe(df[['Ticker','Recommendation','Confidence (%)','Risk Level']].style.applymap(color_risk), width=1000, height=400)

    # Detailed view
    st.markdown("---")
    st.subheader("Detailed Stock Analysis")
    for i, row in df.iterrows():
        with st.expander(f"{row['Ticker']} - {row['Recommendation']} ({row['Confidence (%)']}%)"):
            st.write("Risk Level:", row['Risk Level'])
            st.write("Debate Transcript:\n", row['Debate Transcript'])
            st.write("Entry/Exit Levels:\n", row['Entry/Exit Levels'])

    if st.button("Export CSV"):
        df.to_csv("stock_recommendations.csv", index=False)
        st.success("Report saved as stock_recommendations.csv")
