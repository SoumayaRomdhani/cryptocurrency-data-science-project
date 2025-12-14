import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def display_market_correlations(df):
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
    
    # Supprimer les NaN pour les calculs
    df_clean = df_ticker.dropna(subset=['Price_Change', 'Volume', 'Liquidity'])

    # Corrélation prix-volume
    corr_price_volume = df_clean['Price_Change'].corr(df_clean['Volume'])
    # Corrélation prix-liquidité
    corr_price_liquidity = df_clean['Price_Change'].corr(df_clean['Liquidity'])

    # Graphique de dispersion prix vs volume et prix vs liquidité
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))

    # Prix vs Volume
    ax1.scatter(df_clean['Volume'], df_clean['Price_Change'] * 100, 
                alpha=0.7, color='#667eea', s=25, edgecolors='white', linewidth=0.5)
    # Ligne de tendance
    if len(df_clean) > 1:
        z = np.polyfit(df_clean['Volume'], df_clean['Price_Change'] * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_clean['Volume'].min(), df_clean['Volume'].max(), 100)
        ax1.plot(x_line, p(x_line), color='#e53e3e', linewidth=1.5, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Volume', fontsize=7, color='#4a5568')
    ax1.set_ylabel('Var. Prix (%)', fontsize=7, color='#4a5568')
    ax1.set_title(f'Prix vs Volume (r={corr_price_volume:.2f})', fontsize=8, fontweight='bold', color='#2d3748')
    ax1.tick_params(labelsize=6, colors='#4a5568')
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#e2e8f0')
    ax1.spines['bottom'].set_color('#e2e8f0')
    ax1.grid(True, alpha=0.4, linestyle='--', color='#e2e8f0')

    # Prix vs Liquidité
    ax2.scatter(df_clean['Liquidity'], df_clean['Price_Change'] * 100, 
                alpha=0.7, color='#48bb78', s=25, edgecolors='white', linewidth=0.5)
    # Ligne de tendance
    if len(df_clean) > 1:
        z2 = np.polyfit(df_clean['Liquidity'], df_clean['Price_Change'] * 100, 1)
        p2 = np.poly1d(z2)
        x_line2 = np.linspace(df_clean['Liquidity'].min(), df_clean['Liquidity'].max(), 100)
        ax2.plot(x_line2, p2(x_line2), color='#e53e3e', linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Liquidité', fontsize=7, color='#4a5568')
    ax2.set_ylabel('Var. Prix (%)', fontsize=7, color='#4a5568')
    ax2.set_title(f'Prix vs Liquidité (r={corr_price_liquidity:.2f})', fontsize=8, fontweight='bold', color='#2d3748')
    ax2.tick_params(labelsize=6, colors='#4a5568')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#e2e8f0')
    ax2.spines['bottom'].set_color('#e2e8f0')
    ax2.grid(True, alpha=0.4, linestyle='--', color='#e2e8f0')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Affichage des coefficients de corrélation
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Corrélation Prix-Volume", f"{corr_price_volume:.3f}")
    with col2:
        st.metric("Corrélation Prix-Liquidité", f"{corr_price_liquidity:.3f}")