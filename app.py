import streamlit as st
import pandas as pd

from descriptif.price_evolution import display_price_evolution
from descriptif.volume_analysis import display_volume_analysis
from descriptif.heatmap_correlation import corrélation_heatmap
from descriptif.classement_domination import display_classement_domination 

# Chargement des données nettoyées
df = pd.read_csv('data/données_historique_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])

df_snapshot = pd.read_csv('data/crypto_clean.csv')


# Appel des fonctions depuis les fichiers descriptif
display_price_evolution(df)
display_volume_analysis(df)
corrélation_heatmap(df)
display_classement_domination(df_snapshot)
