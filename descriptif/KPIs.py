import streamlit as st
import pandas as pd
import numpy as np


def display_universe_kpis(df: pd.DataFrame, universe_size: int = 9) -> None:

    st.subheader("KPIs")

    data = df.copy()

    for col in ["Market Cap", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # On enlève les lignes sans Market Cap 
    data = data.dropna(subset=["Market Cap"])
    data["Volume"] = data["Volume"].fillna(0)

    if data.empty:
        st.warning("Pas de cryptos avec Market Cap valide.")
        return

    data = data.sort_values("Market Cap", ascending=False)

    top_n = min(universe_size, len(data))
    universe = data.head(top_n).reset_index(drop=True)

    if top_n < universe_size:
        st.info(f"Seulement {top_n} cryptos ont une Market Cap valide dans le dataset.")

    symbols = universe["Symbol"].tolist()


    symbol = st.selectbox("Choisissez une crypto :", symbols)
    coin = universe[universe["Symbol"] == symbol].iloc[0]

   
    total_mcap = universe["Market Cap"].sum()
    total_vol = universe["Volume"].sum()

    

    # a) Market dominance 
    market_dom = coin["Market Cap"] / total_mcap if total_mcap > 0 else np.nan

    # b) Volume dominance 
    volume_dom = coin["Volume"] / total_vol if total_vol > 0 else np.nan

    # c) Turnover Speed 
    if coin["Volume"] > 0:
        turnover_days = coin["Market Cap"] / coin["Volume"]
    else:
        turnover_days = np.nan

    #dispaly
    col1, col2, col3 = st.columns(3)

    col1.metric(
        " Market dominance",
        "N/A" if pd.isna(market_dom) else f"{market_dom * 100:.1f} %",
        help="Part de cette crypto dans la capitalisation totale de l'univers."
    )

    col2.metric(
        "Volume dominance",
        "N/A" if pd.isna(volume_dom) else f"{volume_dom * 100:.1f} %",
        help="Part du volume total représentée par cette crypto."
    )

    col3.metric(
        "Turnover Speed",
        "N/A" if pd.isna(turnover_days) else f"{turnover_days:.1f} j",
        help="Nombre de jours nécessaires pour échanger l'équivalent de toute la Market Cap."
    )

    

