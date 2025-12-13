import streamlit as st
import matplotlib.pyplot as plt

def display_volume_analysis(df):
    st.subheader("2. Analyse des Volumes")

    # Filtrer les données pour les mois de septembre, octobre, novembre
    df_filtered = df[df['Date'].dt.month.isin([9, 10, 11])]

    for ticker in df_filtered['Ticker'].unique():
        df_ticker = df_filtered[df_filtered['Ticker'] == ticker].sort_values('Date')
        plt.figure(figsize=(10, 5))
        plt.bar(df_ticker['Date'], df_ticker['Volume'])
        plt.title(f"Analyse des Volumes - {ticker} (Sep-Nov)")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.xticks(rotation=45)
        st.pyplot(plt)