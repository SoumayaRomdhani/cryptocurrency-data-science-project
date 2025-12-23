import streamlit as st
import pandas as pd
import numpy as np


class UniverseKPIs:
    """
    KPIs univers (top N cryptos par Market Cap):
    - selectbox pour choisir une crypto
    - 3 KPI cards: Market dominance, Volume dominance, Turnover speed
    """

    def __init__(self, default_universe_size: int = 9) -> None:
        self.default_universe_size = default_universe_size

    @staticmethod
    def _inject_css() -> None:
        # CSS identique à ton fichier original (garde le rendu)
        st.markdown(
            """
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,700,1,200');

.ms{
  font-family: 'Material Symbols Rounded';
  font-weight: 700;
  font-style: normal;
  font-size: 20px;
  line-height: 1;
  display: inline-block;
  text-transform: none;
  letter-spacing: normal;
  white-space: nowrap;
  direction: ltr;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  -moz-osx-font-smoothing: grayscale;
  font-variation-settings: "FILL" 1, "wght" 700, "GRAD" 200, "opsz" 24;
  color: rgba(0,0,0,0.82) !important;
  filter: drop-shadow(0 8px 16px rgba(0,0,0,0.18)) !important;
}

div[data-testid="stSelectbox"] label{
  font-weight: 950 !important;
  letter-spacing: .2px;
  font-size: 0.95rem !important;
  opacity: .92;
  margin-bottom: .40rem;
}

div[data-testid="stSelectbox"]{
  max-width: 640px;
}
div[data-testid="stSelectbox"] [data-baseweb="select"]{
  max-width: 640px;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  width: 100% !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  border-radius: 18px !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  background: rgba(255,255,255,0.065) !important;
  backdrop-filter: blur(14px) !important;
  -webkit-backdrop-filter: blur(14px) !important;
  box-shadow: 0 16px 40px rgba(0,0,0,0.24) !important;
  min-height: 48px !important;
  padding-left: 10px !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] span{
  max-width: 520px !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover{
  border-color: rgba(255,255,255,0.32) !important;
  background: rgba(255,255,255,0.085) !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within{
  border-color: rgba(99,102,241,0.75) !important;
  box-shadow:
    0 0 0 5px rgba(99,102,241,0.17),
    0 16px 40px rgba(0,0,0,0.24) !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] svg{
  opacity: .78 !important;
  transform: translateY(1px);
}

[data-baseweb="popover"] [data-baseweb="menu"]{
  background: rgba(18, 18, 24, 0.78) !important;
  backdrop-filter: blur(18px) !important;
  -webkit-backdrop-filter: blur(18px) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  box-shadow: 0 22px 55px rgba(0,0,0,0.45) !important;
  border-radius: 16px !important;
  padding: .45rem !important;
}

[data-baseweb="popover"] [role="option"]{
  border-radius: 14px !important;
  margin: 6px 0 !important;
  padding: .62rem .75rem !important;
  font-weight: 650 !important;
  letter-spacing: .15px !important;
}

[data-baseweb="popover"] [role="option"]:hover{
  background: rgba(255,255,255,0.11) !important;
}

[data-baseweb="popover"] [aria-selected="true"]{
  background: rgba(99,102,241,0.24) !important;
  border: 1px solid rgba(99,102,241,0.40) !important;
}

.kpi-card{
  border-radius: 18px;
  padding: 1.0rem 1.05rem;
  border: 1px solid rgba(255,255,255,0.18);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 16px 40px rgba(0,0,0,0.25);
  position: relative;
  overflow: hidden;
}

.kpi-card:before{
  content:"";
  position:absolute; inset:-60px;
  background: radial-gradient(circle at 25% 20%, rgba(255,255,255,0.20), transparent 55%);
  pointer-events:none;
}

.kpi-purple{
  background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(168,85,247,0.10));
  border-color: rgba(99,102,241,0.30);
}
.kpi-cyan{
  background: linear-gradient(135deg, rgba(34,211,238,0.16), rgba(59,130,246,0.10));
  border-color: rgba(34,211,238,0.28);
}
.kpi-amber{
  background: linear-gradient(135deg, rgba(245,158,11,0.16), rgba(239,68,68,0.10));
  border-color: rgba(245,158,11,0.28);
}

.kpi-head{
  display:flex; align-items:center; justify-content:space-between;
  gap:.75rem; margin-bottom:.55rem;
}
.kpi-title{
  font-size: 0.98rem;
  font-weight: 900;
  margin: 0;
  letter-spacing: .2px;
  opacity: .95;
}
.kpi-value{
  font-size: 2.35rem;
  font-weight: 950;
  margin: 0.05rem 0 0 0;
  line-height: 1.05;
}
.kpi-caption{
  margin-top: 0.55rem;
  font-size: 0.86rem;
  opacity: 0.80;
  line-height: 1.35;
}

.icon-chip{
  width: 40px; height: 40px;
  border-radius: 14px;
  display:flex; align-items:center; justify-content:center;
  border: 1px solid rgba(255,255,255,0.20);
  box-shadow: 0 10px 24px rgba(0,0,0,0.20);
}

.chip-purple{
  background: linear-gradient(135deg, rgba(99,102,241,0.48), rgba(168,85,247,0.32));
  box-shadow: 0 10px 26px rgba(99,102,241,0.22);
}
.chip-cyan{
  background: linear-gradient(135deg, rgba(34,211,238,0.44), rgba(59,130,246,0.30));
  box-shadow: 0 10px 26px rgba(34,211,238,0.20);
}
.chip-amber{
  background: linear-gradient(135deg, rgba(245,158,11,0.44), rgba(239,68,68,0.28));
  box-shadow: 0 10px 26px rgba(245,158,11,0.18);
}
</style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for col in ["Market Cap", "Volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=["Market Cap"])
        data["Volume"] = data["Volume"].fillna(0)

        return data

    def render(self, df: pd.DataFrame, universe_size: int | None = None) -> None:
        universe_size = self.default_universe_size if universe_size is None else universe_size

        data = self._clean(df)

        if data.empty:
            st.warning("Pas de cryptos avec Market Cap valide.")
            return

        data = data.sort_values("Market Cap", ascending=False)
        top_n = min(universe_size, len(data))
        universe = data.head(top_n).reset_index(drop=True)

        if top_n < universe_size:
            st.info(f"Seulement {top_n} cryptos ont une Market Cap valide dans le dataset.")

        symbols = universe["Symbol"].tolist()

        self._inject_css()

        symbol = st.selectbox("Choisissez une crypto :", symbols)
        coin = universe[universe["Symbol"] == symbol].iloc[0]

        total_mcap = universe["Market Cap"].sum()
        total_vol = universe["Volume"].sum()

        market_dom = coin["Market Cap"] / total_mcap if total_mcap > 0 else np.nan
        volume_dom = coin["Volume"] / total_vol if total_vol > 0 else np.nan
        turnover_days = (coin["Market Cap"] / coin["Volume"]) if coin["Volume"] > 0 else np.nan

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            value = "N/A" if pd.isna(market_dom) else f"{market_dom * 100:.1f} %"
            st.markdown(
                f"""
                  <div class="kpi-card kpi-purple">
                    <div class="kpi-head">
                      <p class="kpi-title">Market Dominance</p>
                      <div class="icon-chip chip-purple"><span class="ms">trophy</span></div>
                    </div>
                    <p class="kpi-value">{value}</p>
                    <p class="kpi-caption">la Part de cette crypto dans la capitalisation totale de l’univers.</p>
                  </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            value = "N/A" if pd.isna(volume_dom) else f"{volume_dom * 100:.1f} %"
            st.markdown(
                f"""
                  <div class="kpi-card kpi-cyan">
                    <div class="kpi-head">
                      <p class="kpi-title">Volume Dominance</p>
                      <div class="icon-chip chip-cyan"><span class="ms">query_stats</span></div>
                    </div>
                    <p class="kpi-value">{value}</p>
                    <p class="kpi-caption">la part du volume total représentée par cette crypto.</p>
                  </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            value = "N/A" if pd.isna(turnover_days) else f"{turnover_days:.1f} j"
            st.markdown(
                f"""
                <div class="kpi-card kpi-amber">
                  <div class="kpi-head">
                    <p class="kpi-title">Turnover Speed</p>
                    <div class="icon-chip chip-amber"><span class="ms">speed</span></div>
                  </div>
                  <p class="kpi-value">{value}</p>
                  <p class="kpi-caption">Nombre de jours nécessaires pour échanger l’équivalent de toute la Market Cap.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
