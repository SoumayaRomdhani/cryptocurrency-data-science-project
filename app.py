import streamlit as st
import pandas as pd

from descriptif.price_evolution import PriceEvolution
from descriptif.volume_analysis import VolumeAnalysis
from descriptif.heatmap_correlation import CorrelationHeatmap
from descriptif.classement_domination import ClassementDomination
from descriptif.KPIs import UniverseKPIs
from descriptif.correlation_market_factors import MarketCorrelations
from descriptif.seasonality_analysis import SeasonalityAnalysis
from descriptif.acp_snapshot import ACPSnapshot

from predictif.lstm_model import ModelResultsViewer
from predictif.kmeans_clustering import KMeansClustering


st.set_page_config(
    page_title="CryptoCurrencyTracker Dashboard",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
:root{
  --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  --secondary-gradient: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
  --card-shadow: 0 6px 14px rgba(0,0,0,0.08);
  --hover-shadow: 0 14px 26px rgba(0,0,0,0.14);
  --border-radius: 14px;
  --text-primary: #1f2937;
  --text-secondary: #4b5563;
  --accent: #667eea;
  --accent2: #764ba2;
}

/* Page */
.main .block-container{
  padding: 2rem 3rem;
  max-width: 1400px;
}

/* ===================== BEAUTIFUL BRAND TITLE ===================== */
.brand-wrap{
  padding: 0.75rem 0.75rem 1.0rem 0.75rem;
  margin: 0.25rem 0.75rem 0.9rem 0.75rem;
}
.brand-title{
  margin: 0;
  text-align: center;
  font-weight: 900;
  font-size: 1.65rem;
  letter-spacing: 0.8px;
  background: var(--primary-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
}
.brand-sub{
  margin: 0.25rem 0 0 0;
  text-align: center;
  font-size: 0.85rem;
  color: rgba(31,41,55,0.65);
}
.sidebar-sep{
  height: 1px;
  background: rgba(226,232,240,0.95);
  margin: 0.6rem 0.9rem 1rem 0.9rem;
}

/* ===================== MAIN HEADER ===================== */
.main-header{
  font-size: 2.85rem;
  font-weight: 850;
  background: var(--primary-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-align: center;
  margin-bottom: 2rem;
  padding: 1.2rem 0.6rem;
  letter-spacing: -0.6px;
}

/* ===================== SECTION HEADERS ===================== */
.section-header{
  font-size: 1.55rem;
  font-weight: 750;
  color: var(--text-primary);
  margin-top: 2rem;
  margin-bottom: 1rem;
  padding-bottom: 0.55rem;
  border-bottom: 3px solid var(--accent);
}

/* ===================== METRIC CARDS ===================== */
div[data-testid="metric-container"]{
  background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%);
  border: 1px solid rgba(226,232,240,0.95);
  padding: 1.4rem 1.2rem;
  border-radius: var(--border-radius);
  box-shadow: var(--card-shadow);
  transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
}
div[data-testid="metric-container"]:hover{
  transform: translateY(-4px);
  box-shadow: var(--hover-shadow);
  border-color: rgba(102,126,234,0.35);
}

/* ===================== BUTTONS ===================== */
.stButton > button{
  width: 100%;
  background: var(--primary-gradient);
  color: white;
  border: none;
  padding: 0.78rem 1.2rem;
  border-radius: 10px;
  font-weight: 700;
  font-size: 0.96rem;
  transition: transform 220ms ease, box-shadow 220ms ease, filter 220ms ease;
  box-shadow: 0 4px 12px rgba(102,126,234,0.25);
}
.stButton > button:hover{
  transform: translateY(-2px);
  filter: brightness(1.03);
  box-shadow: 0 10px 22px rgba(102,126,234,0.32);
}

/* ===================== TABS ===================== */
.stTabs [data-baseweb="tab-list"]{
  gap: 8px;
  background-color: rgba(247,250,252,0.9);
  padding: 0.5rem;
  border-radius: 12px;
  border: 1px solid rgba(226,232,240,0.9);
}
.stTabs [data-baseweb="tab"]{
  background-color: white;
  border-radius: 10px;
  padding: 0.7rem 1.3rem;
  font-weight: 700;
  border: 2px solid transparent;
  transition: transform 200ms ease, box-shadow 200ms ease, background 200ms ease;
}
.stTabs [data-baseweb="tab"]:hover{
  transform: translateY(-1px);
  box-shadow: 0 10px 18px rgba(0,0,0,0.07);
}
.stTabs [aria-selected="true"]{
  background: var(--primary-gradient) !important;
  color: white !important;
  border-color: rgba(102,126,234,0.35) !important;
}

/* ===================== SIDEBAR ===================== */
[data-testid="stSidebar"]{
  background: var(--secondary-gradient);
  border-right: 1px solid rgba(226,232,240,0.9);
}
[data-testid="stSidebar"] .block-container{
  padding-top: 1.25rem;
}

/* Remove default radio label spacing */
section[data-testid="stSidebar"] label[for^="radio"]{
  display: none !important;
}

/* Radiogroup spacing */
section[data-testid="stSidebar"] div[role="radiogroup"]{
  gap: 10px;
}

/* ---- NO ENCLOSED/FRAMED LOOK ---- */
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
  position: relative;
  display: flex;
  align-items: center;
  padding: 0.55rem 0.9rem;
  border-radius: 12px;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  cursor: pointer;
  transition: background 180ms ease, transform 180ms ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{
  background: rgba(102,126,234,0.10) !important;
  transform: translateX(2px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{
  display: none !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label span{
  font-weight: 750;
  color: var(--text-primary);
  font-size: 1.05rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
  background: rgba(102,126,234,0.12) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) span{
  color: #111827;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::after{
  content: "";
  position: absolute;
  left: 10px;
  right: 10px;
  bottom: 4px;
  height: 3px;
  border-radius: 999px;
  background: var(--primary-gradient);
  box-shadow: 0 10px 18px rgba(102,126,234,0.18);
}

/* ===================== INFO CARDS ===================== */
.info-card{
  background: white;
  padding: 1.5rem;
  border-radius: var(--border-radius);
  box-shadow: var(--card-shadow);
  border-left: 4px solid var(--accent);
  margin-bottom: 1rem;
  transition: transform 180ms ease, box-shadow 180ms ease;
}
.info-card:hover{
  transform: translateX(4px);
  box-shadow: var(--hover-shadow);
}

/* Charts */
.stPlotlyChart, .stPyplot{
  background: white;
  padding: 1rem;
  border-radius: var(--border-radius);
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* Welcome banner */
.welcome-banner{
  background: linear-gradient(135deg, rgba(102,126,234,0.10) 0%, rgba(118,75,162,0.10) 100%);
  padding: 2rem;
  border-radius: 16px;
  margin-bottom: 2rem;
  border: 1px solid rgba(226,232,240,0.95);
}
.welcome-banner h2{ color: var(--text-primary); margin-bottom: 0.8rem; }
.welcome-banner p{ font-size: 1.08rem; color: var(--text-secondary); line-height: 1.8; }
</style>
""",
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/données_historique_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_snapshot_data():
    return pd.read_csv("data/crypto_clean.csv")


df = load_data()
df_snapshot = load_snapshot_data()

price_view = PriceEvolution(months=[9, 10, 11], figsize=(5, 2.5))
volume_view = VolumeAnalysis(months=[9, 10, 11], figsize=(5, 2.5))

kpis_view = UniverseKPIs(default_universe_size=9)
acp_snapshot_view = ACPSnapshot(n_components=3, top_n_labels=10, ticker_col="Ticker")

classement_domination_view = ClassementDomination(default_max_coins=9)
heatmap_view = CorrelationHeatmap(months=[9, 10, 11], figsize=(5, 4))
market_corr_view = MarketCorrelations(ticker="BTC-USD", months=[9, 10, 11], figsize=(6, 2.5))

seasonality_view = SeasonalityAnalysis(ticker="BTC-USD", months=[9, 10, 11], figsize=(6, 3))

kmeans_view = KMeansClustering(target_variance=0.90, min_keep=0.80, max_keep=0.95, default_max_k=6, use_log=True)


model_results_view = ModelResultsViewer(default_pkl_path="data/model_results.pkl")


st.sidebar.markdown(
    """
    <div class="brand-wrap">
      <p class="brand-title">CRYPTO TRACKER</p>
      <p class="brand-sub">Analytics • Insights • AI</p>
    </div>
    <div class="sidebar-sep"></div>
    """,
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Menu",
    ["Accueil", "Analyse Descriptive", "Analyse Prédictive"],
    label_visibility="collapsed"
)


if menu == "Accueil":
    st.markdown('<div class="main-header">Bienvenue sur CryptoCurrencyTracker</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="welcome-banner">
            <h2>Dashboard d'Analyse Cryptomonnaie</h2>
            <p>
                Ce dashboard vous offre une vue complète et professionnelle sur le marché des cryptomonnaies.
                Explorez les analyses statistiques avancées et les prédictions basées sur l'intelligence artificielle.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Fonctionnalités")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">Analyse Descriptive</h4>
                <ul style="color: #4a5568; line-height: 1.8;">
                    <li>KPIs et métriques clés du marché</li>
                    <li>Évolution temporelle des prix</li>
                    <li>Analyse des volumes de transaction</li>
                    <li>Matrices de corrélation</li>
                    <li>Analyse saisonnière et tendances</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h4 style="color: #764ba2; margin-bottom: 0.5rem;">Prédictions IA</h4>
                <ul style="color: #4a5568; line-height: 1.8;">
                    <li>Modèle LSTM (Deep Learning)</li>
                    <li>Modèle XGBoost (Gradient Boosting)</li>
                    <li>Clustering K-Means</li>
                    <li>Prédictions de prix futures</li>
                    <li>Comparaison des performances</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

elif menu == "Analyse Descriptive":
    st.markdown('<div class="main-header">Analyse Descriptive du Marché Crypto</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["KPIs & Évolutions", "ACP", "Corrélations", "Saisonnalité"])

    with tab1:
        st.markdown('<div class="section-header">KPIs Globaux</div>', unsafe_allow_html=True)
        kpis_view.render(df_snapshot)

        st.markdown('<div class="section-header">Évolution des Prix</div>', unsafe_allow_html=True)
        price_view.render(df)

        st.markdown('<div class="section-header">Analyse des Volumes</div>', unsafe_allow_html=True)
        volume_view.render(df)

    with tab2:
        st.markdown('<div class="section-header">Analyse en Composantes Principales</div>', unsafe_allow_html=True)
        acp_snapshot_view.render(df_snapshot)

    with tab3:
        st.markdown('<div class="section-header">Heatmap de Corrélation</div>', unsafe_allow_html=True)
        heatmap_view.render(df)

        st.markdown('<div class="section-header">Marché vs Prix (BTC)</div>', unsafe_allow_html=True)
        market_corr_view.render(df)

    with tab4:
        st.markdown('<div class="section-header">Analyse Saisonnière</div>', unsafe_allow_html=True)
        seasonality_view.render(df)

        st.markdown('<div class="section-header">Classement et Domination</div>', unsafe_allow_html=True)
        classement_domination_view.render(df_snapshot)

elif menu == "Analyse Prédictive":
    st.markdown('<div class="main-header">Prédictions de Prix</div>', unsafe_allow_html=True)

    pred_tab1, pred_tab2 = st.tabs(["Clustering K-Means", "Modèles de Régression"])

    with pred_tab1:
        st.markdown('<div class="section-header">Clustering K-Means</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color: #4a5568; margin-bottom: 1rem;">
                Pipeline : StandardScaler → PCA (auto 80–95%) → K-means (K auto via silhouette)
            </p>
            """,
            unsafe_allow_html=True
        )
        kmeans_view.render(df_snapshot)

    with pred_tab2:
        st.markdown('<div class="section-header">Modèles de Prédiction</div>', unsafe_allow_html=True)

        lstm_tab, xgb_tab = st.tabs(["LSTM (Deep Learning)", "XGBoost (Gradient Boosting)"])

        with lstm_tab:
            st.markdown(
                """
                <p style="color: #4a5568; margin-bottom: 1rem;">
                    Réseau de neurones récurrent pour la prédiction de séries temporelles.
                </p>
                """,
                unsafe_allow_html=True
            )

            model_results_view.render(model_filter="lstm")

        with xgb_tab:
            st.markdown(
                """
                <p style="color: #4a5568; margin-bottom: 1rem;">
                    Algorithme d'ensemble basé sur les arbres de décision.
                </p>
                """,
                unsafe_allow_html=True
            )

            model_results_view.render(model_filter="xgb")
