import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def corrélation_heatmap(df):
    df_filtered = df[df['Date'].dt.month.isin([9, 10, 11])]
    corr_matrix = df_filtered.pivot_table(index='Date', columns='Ticker', values='Close')
    corr = corr_matrix.corr()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.7},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Corrélation des Prix (Sep-Nov)", fontsize=9, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)