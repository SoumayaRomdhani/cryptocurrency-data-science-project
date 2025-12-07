import streamlit as st
import matplotlib.pyplot as plt

def corrélation_heatmap(df):
    st.subheader("3. Carte de Chaleur de la Corrélation")
    corr = df.pivot_table(index='Date', columns='Ticker', values='Close').corr()
    
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Carte de Chaleur de la Corrélation des Prix de Clôture")
    
    st.pyplot(plt)