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
    df_ticker['Returns'] = df_ticker['Close'].pct_change()
    df_ticker['Day_of_Week'] = df_ticker['Date'].dt.day_name()
    weekly_stats = df_ticker.groupby('Day_of_Week')['Returns'].agg(['mean', 'std', 'count'])
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_stats = weekly_stats.reindex(order)

    # Boxplot des rendements quotidiens par jour de la semaine
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_day = [df_ticker[df_ticker['Day_of_Week'] == day]['Returns'] * 100 for day in order]
    box = ax.boxplot(data_by_day, labels=['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'], patch_artist=True)

    # Couleurs sobres
    for patch in box['boxes']:
        patch.set_facecolor('#9ecae1')
        patch.set_edgecolor('#3182bd')
    for whisker in box['whiskers']:
        whisker.set_color('#3182bd')
    for cap in box['caps']:
        cap.set_color('#3182bd')
    for median in box['medians']:
        median.set_color('#de2d26')

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Rendements journaliers (%)')
    ax.set_title('Distribution des rendements par jour (Sep-Nov)')
    plt.tight_layout()
    st.pyplot(fig)

    # Afficher le jour avec le meilleur rendement
    best_day = weekly_stats['mean'].idxmax()
    best_return = weekly_stats['mean'].max() * 100

    st.write(f"**Jour avec le meilleur rendement moyen : {best_day} ({best_return:.3f}%)**")