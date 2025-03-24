import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
from datetime import date
from prophet import Prophet
from bokeh.models.widgets import Div
import pytz
import os

# Configure environment
warnings.filterwarnings("ignore")
os.environ['TZ'] = 'America/Sao_Paulo'

try:
    import yfinance as yfin
    #yfin.pdr_override()
except ImportError:
    st.error("Please install yfinance: pip install yfinance")

try:
    import investpy as inv
except ImportError:
    st.error("Please install investpy: pip install investpy")

# Cached functions
@st.cache_data(ttl=3600)
def get_cached_tickers():
    return get_ticker()

@st.cache_data(ttl=3600)
def get_ticker():
    """Get list of Brazilian stock tickers"""
    try:
        br = inv.stocks.get_stocks(country='brazil')
        return [f"{ticker}.SA" for ticker in br.symbol]
    except Exception as e:
        st.warning(f"Couldn't fetch tickers from investpy: {str(e)}")
        return ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA']

def validate_ticker(ticker):
    """Check if a ticker is valid and has data"""
    try:
        yf = yfin.Ticker(ticker)
        if not yf.info:
            return False
        hist = yf.history(period="1mo")
        return not hist.empty
    except:
        return False

def predict_stock(ticker):
    """Make predictions for a given stock ticker"""
    try:
        yf = yfin.Ticker(ticker)
        hist = yf.history(period="max")
        
        if hist.empty:
            st.error(f"No historical data found for {ticker}")
            return None, None
            
        # Prepare data for Prophet
        hist = hist[['Close']].reset_index()
        hist = hist.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Handle timezone - convert to B3 timezone then remove tzinfo
        hist['ds'] = hist['ds'].dt.tz_convert('America/Sao_Paulo').dt.tz_localize(None)
        
        # Remove any NaN values
        hist = hist.dropna()
        
        # Initialize and fit model
        m = Prophet(daily_seasonality=True)
        m.fit(hist)
        
        # Make future dataframe and predictions
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        
        return forecast, m, hist
        
    except Exception as e:
        st.error(f"Error processing {ticker}: {str(e)}")
        return None, None, None

def plot_results(ticker, forecast, model, hist):
    """Display the forecast results and components"""
    if forecast is None or model is None:
        st.error("No forecast data available")
        return
        
    try:
        # Show date range
        first_date = str(hist.ds.min()).split(' ')[0]
        last_date = str(hist.ds.max()).split(' ')[0]
        
        st.markdown("### Historical Data Period")
        st.write(f"{first_date} to {last_date}")
        
        # Plot forecast
        st.markdown(f"### {ticker} - Price Forecast (Next 365 Days)")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
        
        # Plot components
        st.markdown("### Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error generating plots: {str(e)}")

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(page_title="B3 Stock Predictions", layout="wide")
    
    # Header
    st.markdown("""
    <div style="background-color:tomato;padding:10px">
        <h1 style='text-align:center;color:white;'>Brazilian Stock Market Predictions</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    #logo = Image.open('Logo_B3.png')
    #st.sidebar.image(logo, caption="", use_container_width=True)
    activities = ["Predictions", "About"]
    choice = st.sidebar.radio("Menu", activities)
    
    if choice == "Predictions":
        st.markdown("### Select a Stock Ticker")
        
        # Get tickers with loading indicator
        with st.spinner("Loading available tickers..."):
            tickers = get_cached_tickers()
        
        # Filter valid tickers
        valid_tickers = [t for t in tickers if validate_ticker(t)]
        selected_ticker = st.selectbox("Choose a stock", valid_tickers, index=0)
        
        if st.button("Predict Next 365 Days"):
            with st.spinner("Processing stock data and making predictions..."):
                forecast, model, hist = predict_stock(selected_ticker)
                
                if forecast is not None:
                    plot_results(selected_ticker, forecast, model, hist)
                
    elif choice == "About":
        st.subheader("About This App")
        st.markdown("""
        This application uses the Prophet forecasting tool to predict stock prices 
        for Brazilian companies listed on B3 (Brasil Bolsa Balcão).
        
        **Features:**
        - 365-day price forecasts
        - Trend analysis
        - Weekly and yearly seasonality components
        
        **References:**
        - [Yahoo Finance API](https://analyzingalpha.com/yfinance-python)
        - [Investpy Documentation](https://investpy.readthedocs.io/)
        - [Prophet Documentation](https://facebook.github.io/prophet/)
        """)
        
        st.markdown("---")
        st.subheader("Developer")
        st.write("Silvio Lima")
        
        if st.button("Connect on LinkedIn"):
            js = "window.open('https://www.linkedin.com/in/silviocesarlima/')"
            html = f'<img src onerror="{js}">'
            div = Div(text=html)
            st.bokeh_chart(div)

if __name__ == '__main__':
    main()
