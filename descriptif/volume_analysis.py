import streamlit as st
import matplotlib.pyplot as plt

def display_volume_analysis(df):
    st.subheader("2. Analyse des Volumes")
    for ticker in df['Ticker'].unique():
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date')
        plt.figure(figsize=(10, 5))
        plt.bar(df_ticker['Date'], df_ticker['Volume'])
        plt.title(f"Analyse des Volumes - {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.xticks(rotation=45)
        st.pyplot(plt)