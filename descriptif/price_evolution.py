import streamlit as st
import matplotlib.pyplot as plt

def display_price_evolution(df):
    st.subheader("1. Évolution des Prix")

    # Filtrer les données pour les mois de septembre, octobre, novembre
    df_filtered = df[df['Date'].dt.month.isin([9, 10, 11])]

    for ticker in df_filtered['Ticker'].unique():
        df_ticker = df_filtered[df_filtered['Ticker'] == ticker].sort_values('Date')
        plt.figure(figsize=(10, 5))
        plt.plot(df_ticker['Date'], df_ticker['Close'])
        plt.title(f"Évolution des Prix de Clôture - {ticker} (Sep-Nov)")
        plt.xlabel("Date")
        plt.ylabel("Prix de Clôture")
        plt.xticks(rotation=45)
        st.pyplot(plt)