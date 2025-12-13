import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def corrélation_heatmap(df):
    st.subheader("4. Carte de Chaleur de la Corrélation")
    df_filtered = df[df['Date'].dt.month.isin([9, 10, 11])]
    corr_matrix = df_filtered.pivot_table(index='Date', columns='Ticker', values='Close')
    corr = corr_matrix.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Carte de Chaleur de la Corrélation des Prix de Clôture (Sep-Nov)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(plt)