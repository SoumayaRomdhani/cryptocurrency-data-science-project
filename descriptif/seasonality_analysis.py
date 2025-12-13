import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def display_seasonality_analysis(df):
    st.subheader("2. Analyse de la Saisonnalité - Bitcoin")

    # Filtrer les données pour les mois de septembre, octobre, novembre et seulement Bitcoin
    df_filtered = df[(df['Date'].dt.month.isin([9, 10, 11])) & (df['Ticker'] == 'BTC-USD')]

    if df_filtered.empty:
        st.warning("Aucune donnée disponible pour Bitcoin sur la période Sep-Nov")
        return

    df_ticker = df_filtered.sort_values('Date').copy()

    # Calcul des rendements quotidiens
    df_ticker['Returns'] = df_ticker['Close'].pct_change()

    # Extraire jour de la semaine
    df_ticker['Day_of_Week'] = df_ticker['Date'].dt.day_name()

    # Agrégation par jour de la semaine
    weekly_returns = df_ticker.groupby('Day_of_Week')['Returns'].agg(['mean', 'std', 'count'])
    weekly_returns = weekly_returns.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Graphique des rendements par jour de la semaine
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(weekly_returns)), weekly_returns['mean'] * 100, color='skyblue',
                  yerr=weekly_returns['std'] * 100, capsize=5)

    ax.set_xlabel('Jour de la semaine')
    ax.set_ylabel('Rendement moyen (%)')
    ax.set_title('Analyse de la Saisonnalité - Rendements moyens par jour (Sep-Nov)')
    ax.set_xticks(range(len(weekly_returns)))
    ax.set_xticklabels(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'], rotation=45)

    # Ajouter les valeurs sur les barres
    for bar, mean_val in zip(bars, weekly_returns['mean']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                f'{mean_val:.3f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    # Afficher le jour avec le meilleur rendement
    best_day = weekly_returns['mean'].idxmax()
    best_return = weekly_returns['mean'].max() * 100

    st.write(f"**Jour avec le meilleur rendement moyen : {best_day} ({best_return:.3f}%)**")