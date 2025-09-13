import yfinance as yf
import streamlit as st
from transformers import pipeline
import pandas as pd

# Sentiment Analysis Pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Predefined Stock Tickers
tickers = ['VBL.NS', 'DABUR.NS', 'PFC.NS', 'ABDL.NS', 'JIO.NS', 'FINANCE.NS']

def fetch_stock_data(ticker):
    data = yf.Ticker(ticker)
    hist = data.history(period="3mo")
    info = data.info
    return hist, info

def price_action_agent(hist):
    ma_10 = hist['Close'].rolling(window=10).mean().iloc[-1]
    ma_30 = hist['Close'].rolling(window=30).mean().iloc[-1]
    
    if ma_10 > ma_30:
        return "Bullish trend detected (MA 10 > MA 30)", 0.7
    else:
        return "Bearish or neutral trend (MA 10 <= MA 30)", 0.3

def sentiment_agent(ticker):
    headlines = [
        f"{ticker} shows strong quarterly earnings growth",
        f"{ticker} faces regulatory challenges this quarter"
    ]
    sentiments = [sentiment_analyzer(headline)[0] for headline in headlines]
    positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
    
    if positive_count >= 1:
        return f"Positive news sentiment detected ({positive_count}/2 headlines positive)", 0.6
    else:
        return "Mostly neutral or negative sentiment", 0.4

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
    
    final_score = (pa_score * 0.6)*
