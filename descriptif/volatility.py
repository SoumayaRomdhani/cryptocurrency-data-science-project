import streamlit as st

def display_volatility(df):
    st.subheader("2. Analyse de la Volatilité")
    for ticker in df['Ticker'].unique():
        df_ticker = df[df['Ticker'] == ticker].sort_values('Date')
        df_ticker['Rendements'] = df_ticker['Close'].pct_change()
        volatilite = df_ticker['Rendements'].std()
        st.write(f"Volatilité (écart-type des rendements quotidiens) pour {ticker}: {volatilite:.4f}")