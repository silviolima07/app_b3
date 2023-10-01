
"""app_b3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mq6QSirw-3yPqSniY6ciUa4WRhAP2jiQ
"""

#!pip install streamlit --q

#!pip install pyngrok --q

#!pip install investpy --q

#from pyngrok import ngrok

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# 
import streamlit as st
from PIL import Image
import pandas as pd
import investpy as inv
import yfinance as yfin
yfin.pdr_override()
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")
from datetime import date

#import prophet
from prophet import Prophet

from bokeh.models.widgets import Div

#


def get_ticker():
    br = inv.stocks.get_stocks(country='brazil')
    lista_tickers = []
    sufixo = '.SA'
    for i in [ticker for ticker in br.symbol]:
      lista_tickers.append(i+sufixo)
    return lista_tickers
 
def predict(ticker):
    yf = yfin.Ticker(ticker)
    st.write(yf.info['longName'])
    symbol =  yf.info['symbol']
    description = yf.info['longName']
    #print("predict->Stock: ", yf.info['symbol'])
    #print("predict->Name: " , yf.info['longName'])
    hist = yf.history(period="max")
    
    hist = hist[['Close']]
    hist.reset_index(inplace=True)
    hist = hist.rename({'Date': 'ds', 'Close': 'y'}, axis='columns')
    hist['ds'] = hist['ds'].dt.tz_localize(None)
    #st.write(hist)
    #m = Prophet(daily_seasonality=True)
    m = Prophet()
    m.fit(hist)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    first_date = str(hist.ds.min()).split(' ')[0]
    last_date =  str(hist.ds.max()).split(' ')[0]
    st.write("Period collected : "+first_date, " / "+last_date)
    st.write("forecast")
    st.write(forecast)
    st.write(m)
    return (symbol, description, forecast,m)  
 
def save_plot(symbol, description,forecast,m):
    #print(symbol)
    #print("Predictions\n\ty(future value)\tTrend\tWeakly\tYearly\tDaily")
    fig1 = m.plot(forecast)
    fig1.savefig('prophetplot1.png')
    st.markdown('### '+symbol+" -> "+description)
    st.markdown("### Prediction values for next 365 days")
    st.image('prophetplot1.png')
    #
    st.markdown("### Components: Trend Weekly Yearly Daily")
    fig2 =  m.plot_components(forecast)
    fig2.savefig("prophetplot2.png")
    st.image('prophetplot2.png')

def predict2(ticker):
    st.write('Funcao  predict2')
    yf = yfin.Ticker(ticker)
    st.write(yf.info)
    #st.write(yf.info['longName'])
    symbol =  yf.info['symbol']
    description = yf.info['longName']
    #print("predict->Stock: ", symbol)
    #print("predict->Name: " , description)
    hist = yf.history(period="max")
    """
    hist = hist[['Close']]
    hist.reset_index(inplace=True)
    hist = hist.rename({'Date': 'ds', 'Close': 'y'}, axis='columns')
    hist['ds'] = hist['ds'].dt.tz_localize(None)
    #st.write(hist)
    #m = Prophet(daily_seasonality=True)
    m = Prophet()
    m.fit(hist)
    #st.write("Model Prophet")
    #st.write(m)
    
    future = m.make_future_dataframe(periods=365)
    
    forecast = m.predict(future)
    #st.write(forecast)
    first_date = str(hist.ds.min()).split(' ')[0]
    last_date =  str(hist.ds.max()).split(' ')[0]
    st.write("Period collected : "+first_date, " / "+last_date)
    
    #st.write("Model Prophet")
    #st.write(m)
    
    #st.write("Doing Forecast")
    #st.write(forecast)
    return (symbol, description, forecast,m) 
    """
def main():
 
 
    """B3 App """
     
    html_page = """
     <div style="background-color:tomato;padding=50px">
         <p style='text-align:center;font-size:50px;font-weight:bold'>Stocks & Prophet</p>
     </div>
               """
    st.markdown(html_page, unsafe_allow_html=True)
 
    image = Image.open("Logo_B3 .png")
    st.sidebar.image(image,caption="", use_column_width=True)
    
    activities = ["Predictions","About"]
    choice = st.sidebar.radio("Home",activities)
    
    if choice == 'Predictions':
        #predict('TAEE4.SA') 
        st.markdown("### Choose a ticker")
        option = st.selectbox('Ticker',get_ticker(), label_visibility = 'hidden')
        st.write('You selected:', option)
        if st.button("Predicting next 365 DAYS"):
            try:
               #symbol, description,forecast,model = predict2(option)
               #save_plot(symbol, description,forecast,model)
               with st.spinner('Wait for it...we are collecting data'):
                   
                   symbol, description,forecast,model = predict2(option)
                   #save_plot(symbol, description,forecast,model)
            except:
               st.write("Error Ticker: "+option)
               st.error('THIS TICKER WAS LIKELY RENAMED.', icon="🚨")
 
    if choice == 'About':
        st.subheader("I hope you enjoy it.")
        st.markdown("### References:")
        st.markdown("##### - https://analyzingalpha.com/yfinance-python")
        st.markdown("##### - https://analisemacro.com.br/mercado-financeiro/datareader-e-analises-com-yahoo-finance/")
        st.markdown("##### - https://levelup.gitconnected.com/how-to-plot-stock-prices-using-python-87ba684d126c")
        st.markdown("##### - https://www.youtube.com/watch?v=ZxLJjUcP2LI")
        st.markdown("##### - https://www.section.io/engineering-education/how-to-plot-a-candlestick-chart-in-python-using-the-matplotlib-finance-api/")
        st.markdown("##### - https://hareeshpb.medium.com/stock-prediction-using-prophet-python-525710e1ab0c")
        st.subheader("by Silvio Lima")
         
        if st.button("Linkedin"):
             js = "window.open('https://www.linkedin.com/in/silviocesarlima/')"
             html = '<img src onerror="{}">'.format(js)
             div = Div(text=html)
             st.bokeh_chart(div)
 
 
 
if __name__ == '__main__':
    main()



#!nohup streamlit run app.py &

#!streamlit run /content/app.py &>/content/logs.txt &

#ngrok.kill()

# Terminate ngrok port
#ngrok.kill()
# Set authentication (optional)
# Get your authentication token via https://dashboard.ngrok.com/auth
#
#NGROK_AUTH_TOKEN = "1gNjeFx7GPLDxTs5H60p5ZeZgl8_2NQZEtskwajdxd3ge8ZSx"
#ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Open an HTTPs tunnel on port 5000 for http://localhost:5000
#ngrok_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
#print("Streamlit Tracking UI:", ngrok_tunnel.public_url)
