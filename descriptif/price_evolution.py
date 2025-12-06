import streamlit as st
import matplotlib.pyplot as plt

def display_price_evolution(df):
    st.subheader("1. Évolution des Prix")
    for ticker in df['Ticker'].unique():
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date')
        plt.figure(figsize=(10, 5))
        plt.plot(df_ticker['Date'], df_ticker['Close'])
        plt.title(f"Évolution des Prix de Clôture - {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Prix de Clôture")
        plt.xticks(rotation=45)
        st.pyplot(plt)