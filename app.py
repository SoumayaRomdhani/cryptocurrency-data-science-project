import streamlit as st
import pandas as pd

from descriptif.price_evolution import display_price_evolution
from descriptif.volatility import display_volatility
from descriptif.volume_analysis import display_volume_analysis

# Chargement des données nettoyées
df = pd.read_csv('data/données_historique_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])



# Appel des fonctions depuis les fichiers descriptif
display_price_evolution(df)
display_volatility(df)
display_volume_analysis(df)


