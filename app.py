import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# --- Sentiment pipeline: FinBERT for finance ---
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

finbert_tokenizer, finbert_model = load_finbert()

def sentiment_agent_financial(headlines):
    scores = []
    for text in headlines:
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = finbert_model(**inputs)
        probs = softmax(outputs.logits.detach().numpy()[0])
        # probs = [positive, negative, neutral]
        scores.append(probs[0] - probs[1])  # positive minus negative
    if scores:
        avg_score = sum(scores)/len(scores)
        return f"{len([s for s in scores if s>0])}/{len(scores)} positive", max(0, min(0.25 + avg_score*0.25, 1.0))
    else:
        return "No news", 0.1

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

# --- Top stocks by volume ---
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

# --- Agents ---
def price_action_agent(hist):
    ma10 = hist['Close'].rolling(10).mean().iloc[-1]
    ma30 = hist['Close'].rolling(30).mean().iloc[-1]
    last = hist['Close'].iloc[-1]
    if ma10 > ma30:
        return "Bullish MA", 0.3, last*1.01, last*1.05
    else:
        return "Bearish MA", 0.15, last*0.99, last*0.95

def technical_agent_enhanced(hist):
    close = hist['Close']
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    macd_signal = macd.iloc[-1] - signal.iloc[-1]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2*std20
    lower = sma20 - 2*std20
    bollinger_signal = 1 if close.iloc[-1] < upper.iloc[-1] else -1

    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up/roll_down
    rsi = 100 - 100/(1+rs)
    rsi_signal = 1 if rsi.iloc[-1] < 70 else -1

    score = 0.15 + 0.1*(macd_signal>0) + 0.05*(bollinger_signal>0) + 0.05*(rsi_signal>0)
    score = min(score, 1.0)
    return f"MACD={round(macd.iloc[-1],2)}, RSI={round(rsi.iloc[-1],1)}", score, None, None

def fundamental_agent_enhanced(info):
    pe = info.get('trailingPE', None)
    roe = info.get('returnOnEquity', None)
    de = info.get('debtToEquity', None)
    epsg = info.get('earningsQuarterlyGrowth', None)
    div = info.get('dividendYield', None)
    
    score = 0.1
    args = []

    if pe and pe<25: 
        score += 0.05
        args.append(f"PE={pe}")
    if roe and roe>0.15:
        score += 0.05
        args.append(f"ROE={round(roe,2)}")
    if de and de<1:
        score += 0.05
        args.append(f"D/E={de}")
    if epsg and epsg>0.05:
        score += 0.05
        args.append(f"EPSG={round(epsg,2)}")
    if div:
        score += 0.05
        args.append(f"DivYield={round(div,2)}")
    
    score = min(score,1.0)
    return " | ".join(args) if args else "Moderate fundamentals", score, None, None

def volume_agent(hist):
    v10 = hist['Volume'].rolling(10).mean().iloc[-1]
    v30 = hist['Volume'].rolling(30).mean().iloc[-1]
    if v10>1.5*v30:
        return "Volume spike", 0.1, None, None
    else:
        return "Volume normal", 0.1, None, None

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
    debate['Technical'] = technical_agent_enhanced(hist)
    
    # Realistic news headlines via yfinance (limited)
    try:
        news_items = yf.Ticker(ticker).news[:5]
        headlines = [n['title'] for n in news_items]
    except:
        headlines = [f"{ticker} reports strong earnings", f"{ticker} faces regulatory risks"]
    debate['Sentiment'] = sentiment_agent_financial(headlines)
    
    debate['Fundamentals'] = fundamental_agent_enhanced(info)
    debate['Volume'] = volume_agent(hist)

    adjusted = moderator(debate)

    agent_weights = [0.3,0.25,0.25,0.2,0.1]
    max_possible_score = sum(agent_weights)
    final_score = sum(score for _,score,_,_ in adjusted.values())
    final_score_normalized = min(final_score/max_possible_score, 1.0)

    volatility = hist['Close'].pct_change().rolling(30).std().iloc[-1]
    final_score_normalized *= max(0.8,1-volatility)

    confidence_percent = round(final_score_normalized*100,2)
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
st.title("Multi-Agent Stock Recommendation System (Enhanced Phase 6)")

def format_entry_exit(d):
    if not isinstance(d, dict):
        return {}
    formatted = {}
    for k,v in d.items():
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

    def color_risk(val):
        if val=="High-risk": return "background-color: #ff9999"
        elif val=="Medium-risk": return "background-color: #fff799"
        else: return "background-color: #b3ffb3"

    st.dataframe(df[['Ticker','Recommendation','Confidence (%)','Risk Level']].style.applymap(color_risk), width=1000, height=400)

    st.markdown("---")
    st.subheader("Detailed Stock Analysis")
    for i, row in df.iterrows():
        with st.expander(f"{row['Ticker']} - {row['Recommendation']} ({row['Confidence (%)']}%, {row['Risk Level']})"):
            st.text("Debate Transcript:\n"+row['Debate Transcript'])
            st.text("Entry/Exit Levels:\n"+str(row['Entry/Exit Levels']))

    if st.button("Export as CSV"):
        df.to_csv('stock_recommendations_phase6.csv', index=False)
        st.success("Report saved as stock_recommendations_phase6.csv")
