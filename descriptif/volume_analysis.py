import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def display_volume_analysis(df):
    # Filtrer les données pour les mois de septembre, octobre, novembre
    df_filtered = df[df['Date'].dt.month.isin([9, 10, 11])]
    
    tickers = df_filtered['Ticker'].unique()
    colors = ['#764ba2', '#667eea', '#48bb78', '#ed8936', '#e53e3e']
    
    cols = st.columns(2)
    
    for idx, ticker in enumerate(tickers):
        df_ticker = df_filtered[df_filtered['Ticker'] == ticker].sort_values('Date')
        color = colors[idx % len(colors)]
        
        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.bar(df_ticker['Date'], df_ticker['Volume'], color=color, alpha=0.75, width=0.8)
        ax.set_title(f"{ticker}", fontsize=9, fontweight='bold', color='#2d3748')
        ax.set_xlabel("", fontsize=7)
        ax.set_ylabel("Vol.", fontsize=7, color='#4a5568')
        ax.tick_params(axis='x', rotation=45, labelsize=6, colors='#4a5568')
        ax.tick_params(axis='y', labelsize=6, colors='#4a5568')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.grid(True, alpha=0.4, axis='y', linestyle='--', color='#e2e8f0')
        plt.tight_layout()
        
        with cols[idx % 2]:
            st.pyplot(fig)
        plt.close(fig)