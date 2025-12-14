import streamlit as st
import pandas as pd


from descriptif.price_evolution import display_price_evolution
from descriptif.volume_analysis import display_volume_analysis
from descriptif.heatmap_correlation import corrélation_heatmap
from descriptif.classement_domination import display_classement_domination
from descriptif.KPIs import display_universe_kpis
from descriptif.correlation_market_factors import display_market_correlations
from descriptif.seasonality_analysis import display_seasonality_analysis
from descriptif.acp_snapshot import display_acp_snapshot

from predictif.lstm_model import display_model_results
from predictif.kmeans_clustering import display_kmeans_clustering


st.set_page_config(
    page_title="CryptoCurrencyTracker Dashboard",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Root variables for consistent theming */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        --hover-shadow: 0 8px 15px rgba(0, 0, 0, 0.12);
        --border-radius: 12px;
        --text-primary: #2d3748;
        --text-secondary: #4a5568;
        --accent-purple: #667eea;
        --accent-violet: #764ba2;
    }

    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--accent-purple);
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: var(--hover-shadow);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: var(--primary-gradient);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f7fafc;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient);
        color: white;
        border-color: var(--accent-purple);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--secondary-gradient);
    }
    
    [data-testid="stSidebar"] h1 {
        color: var(--text-primary);
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Main content container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--accent-purple);
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .info-card:hover {
        transform: translateX(4px);
    }
    
    /* Chart containers */
    .stPlotlyChart, .stPyplot {
        background: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Welcome banner */
    .welcome-banner {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .welcome-banner h2 {
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    .welcome-banner p {
        font-size: 1.1rem;
        color: var(--text-secondary);
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# Load cleaned data


@st.cache_data
def load_data():
    """Load and preprocess the historical cryptocurrency data."""
    df = pd.read_csv('data/données_historique_cleaned.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@st.cache_data
def load_snapshot_data():
    """Load the cryptocurrency snapshot data."""
    return pd.read_csv('data/crypto_clean.csv')


df = load_data()
df_snapshot = load_snapshot_data()

# Sidebar navigation
st.sidebar.title("CryptoTracker")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")

menu = st.sidebar.radio(
    "Menu",
    ["Accueil", "Dashboard", "Prédictions"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #718096; font-size: 0.85rem;">
        <p>Analysez le marché crypto<br>avec des données en temps réel</p>
    </div>
    """,
    unsafe_allow_html=True
)

if menu == "Accueil":
    st.markdown(
        '<div class="main-header">Bienvenue sur CryptoCurrencyTracker</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="welcome-banner">
        <h2>Dashboard d'Analyse Cryptomonnaie</h2>
        <p>
            Ce dashboard vous offre une vue complète et professionnelle sur le marché des cryptomonnaies.
            Explorez les analyses statistiques avancées et les prédictions basées sur l'intelligence artificielle.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Fonctionnalités")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
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
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
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
        """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("### Aperçu Rapide")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cryptos Analysées", f"{df_snapshot.shape[0]}")
    with col2:
        st.metric("Période Historique", f"{df['Date'].dt.year.nunique()} ans")
    with col3:
        st.metric("Points de Données", f"{len(df):,}")
    with col4:
        st.metric("Dernière MAJ", df['Date'].max().strftime("%d/%m/%Y"))

elif menu == "Dashboard":
    st.markdown(
        '<div class="main-header">Analyse Descriptive du Marché Crypto</div>',
        unsafe_allow_html=True
    )

    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "KPIs & Évolutions",
        "ACP",
        "Corrélations",
        "Saisonnalité"
    ])

    with tab1:
        st.markdown(
            '<div class="section-header">KPIs Globaux</div>',
            unsafe_allow_html=True
        )
        display_universe_kpis(df_snapshot)

        st.markdown(
            '<div class="section-header">Évolution des Prix</div>',
            unsafe_allow_html=True
        )
        display_price_evolution(df)

        st.markdown(
            '<div class="section-header">Analyse des Volumes</div>',
            unsafe_allow_html=True
        )
        display_volume_analysis(df)

    with tab2:
        st.markdown(
            '<div class="section-header">Analyse en Composantes Principales</div>',
            unsafe_allow_html=True
        )
        display_acp_snapshot(df_snapshot)

    with tab3:
        st.markdown(
            '<div class="section-header">Heatmap de Corrélation</div>',
            unsafe_allow_html=True
        )
        corrélation_heatmap(df)

        st.markdown(
            '<div class="section-header">Marché vs Prix (BTC)</div>',
            unsafe_allow_html=True
        )
        display_market_correlations(df)

    with tab4:
        st.markdown(
            '<div class="section-header">Analyse Saisonnière</div>',
            unsafe_allow_html=True
        )
        display_seasonality_analysis(df)

        st.markdown(
            '<div class="section-header">Classement et Domination</div>',
            unsafe_allow_html=True
        )
        display_classement_domination(df_snapshot)


elif menu == "Prédictions":
    st.markdown(
        '<div class="main-header">Prédictions de Prix</div>',
        unsafe_allow_html=True
    )

    pred_tab1, pred_tab2 = st.tabs(["Clustering K-Means", "Modèles de Régression"])

    with pred_tab1:
        st.markdown(
            '<div class="section-header">Clustering K-Means</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <p style="color: #4a5568; margin-bottom: 1rem;">
                Pipeline : StandardScaler → PCA (auto 80–95%) → K-means (K auto via silhouette)
            </p>
            """,
            unsafe_allow_html=True
        )
        display_kmeans_clustering(df_snapshot)

    with pred_tab2:
        st.markdown(
            '<div class="section-header">Modèles de Prédiction</div>',
            unsafe_allow_html=True
        )

        # Sub-tabs for LSTM and XGBoost
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
            display_model_results(model_filter='lstm')

        with xgb_tab:
            st.markdown(
                """
                <p style="color: #4a5568; margin-bottom: 1rem;">
                    Algorithme d'ensemble basé sur les arbres de décision.
                </p>
                """,
                unsafe_allow_html=True
            )
            display_model_results(model_filter='xgb')
