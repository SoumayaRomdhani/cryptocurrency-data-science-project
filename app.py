import streamlit as st
import pandas as pd

from predictif.lstm_model import display_model_results

from descriptif.price_evolution import display_price_evolution
from descriptif.volume_analysis import display_volume_analysis
from descriptif.heatmap_correlation import corrélation_heatmap
from descriptif.classement_domination import display_classement_domination
from descriptif.KPIs import display_universe_kpis
from descriptif.correlation_market_factors import display_market_correlations
from descriptif.seasonality_analysis import display_seasonality_analysis
from predictif.kmeans_clustering import display_kmeans_clustering
from descriptif.acp_snapshot import display_acp_snapshot
# Chargement des données nettoyées
df = pd.read_csv('data/données_historique_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])

df_snapshot = pd.read_csv('data/crypto_clean.csv')

st.subheader("dashboard")

# Appel des fonctions depuis les fichiers descriptif
display_universe_kpis(df_snapshot)
display_price_evolution(df)
display_volume_analysis(df)
corrélation_heatmap(df)
display_classement_domination(df_snapshot)
display_market_correlations(df)
display_seasonality_analysis(df)
display_acp_snapshot(df_snapshot)

# Appel des fonctions depuis les fichiers predictif
display_kmeans_clustering(df_snapshot)
display_model_results()
