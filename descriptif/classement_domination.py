import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def display_classement_domination(df: pd.DataFrame, max_coins: int = 9) -> None:
    st.subheader("4. Classement & Domination du Marché")

   
    data = df.dropna(subset=["Market Cap", "Volume"]).copy()
    data = data[data["Market Cap"] > 0]  
    data["LiquidityScore"] = data["Volume"] / data["Market Cap"]

    main = data.sort_values("Market Cap", ascending=False).head(max_coins)

    symbols = main["Symbol"]                
    mcap_share = main["Market Cap"] / main["Market Cap"].sum()
    vol_share = main["Volume"] / main["Volume"].sum()
    liq_score = main["LiquidityScore"]

    col1, col2 = st.columns(2)

    # Donut Market Cap
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            mcap_share,
            labels=symbols,
            autopct="%1.1f%%",
            startangle=90,
        )
        centre = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(centre)
        ax.axis("equal")
        ax.set_title("Part de marché (Market Cap)")
        st.pyplot(fig, use_container_width=True)

    # Donut Volume 24h
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            vol_share,
            labels=symbols,
            autopct="%1.1f%%",
            startangle=90,
        )
        centre = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(centre)
        ax.axis("equal")
        ax.set_title("Part du flux (Volume 24h)")
        st.pyplot(fig, use_container_width=True)


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(symbols, liq_score, alpha=0.85)
    ax.set_title("Score de liquidité")
    ax.set_ylabel("Volume / Market Cap")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)

    st.pyplot(fig, use_container_width=True)
