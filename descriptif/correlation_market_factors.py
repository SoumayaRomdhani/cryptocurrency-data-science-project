import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def display_market_correlations(df):
    st.subheader("1. Corrélation entre Variables de Marché et Prix - Bitcoin")

    # Filtrer les données pour les mois de septembre, octobre, novembre et seulement Bitcoin
    df_filtered = df[(df['Date'].dt.month.isin([9, 10, 11])) & (df['Ticker'] == 'BTC-USD')]

    if df_filtered.empty:
        st.warning("Aucune donnée disponible pour Bitcoin sur la période Sep-Nov")
        return

    df_ticker = df_filtered.sort_values('Date').copy()

    # Calcul des variations de prix (rendements quotidiens)
    df_ticker['Price_Change'] = df_ticker['Close'].pct_change()

    # Calcul de la liquidité (proxy: volume relatif au prix moyen)
    df_ticker['Liquidity'] = df_ticker['Volume'] / df_ticker['Close']

    # Corrélation prix-volume
    corr_price_volume = df_ticker['Price_Change'].corr(df_ticker['Volume'])
    # Corrélation prix-liquidité
    corr_price_liquidity = df_ticker['Price_Change'].corr(df_ticker['Liquidity'])

    # Graphique de dispersion prix vs volume et prix vs liquidité
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Prix vs Volume
    ax1.scatter(df_ticker['Volume'], df_ticker['Price_Change'] * 100, alpha=0.6, color='blue')
    ax1.set_xlabel('Volume')
    ax1.set_ylabel('Variation Prix (%)')
    ax1.set_title(f'Prix vs Volume\nCorrélation: {corr_price_volume:.3f}')
    ax1.grid(True, alpha=0.3)

    # Prix vs Liquidité
    ax2.scatter(df_ticker['Liquidity'], df_ticker['Price_Change'] * 100, alpha=0.6, color='green')
    ax2.set_xlabel('Liquidité')
    ax2.set_ylabel('Variation Prix (%)')
    ax2.set_title(f'Prix vs Liquidité\nCorrélation: {corr_price_liquidity:.3f}')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Analyse des Corrélations Marché-Prix - Bitcoin (Sep-Nov)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    # Affichage des coefficients de corrélation
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Corrélation Prix-Volume", f"{corr_price_volume:.3f}")
    with col2:
        st.metric("Corrélation Prix-Liquidité", f"{corr_price_liquidity:.3f}")