import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

# We using facebook prophet here from prediction


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of predeiction", 1, 4)
period = n_years * 365

@st.cache # will cash the downloaded data so it wont download again and again
def load_data(ticker):
    data = yf.download(ticker, START, TODAY) # return data in dataframe format
    data.reset_index(inplace = True) # replace index with date as index

    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'stock_close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forcasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date":"ds", "Close":"y"}) # fbprophen requires this format so we rename the df

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods = period)
forcast = model.predict(future)

st.subheader("Forcasted data")
st.write(forcast.tail())

st.write('Forcast data')
forcast_fig = plot_plotly(model, forcast)
st.plotly_chart(forcast_fig)

st.write('Forcast components')
fig2 = model.plot_components(forcast)
st.write(fig2)






#
