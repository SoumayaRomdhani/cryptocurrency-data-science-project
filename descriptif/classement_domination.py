import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def display_classement_domination(df: pd.DataFrame, max_coins: int = 9) -> None:
    st.subheader(" Classement & Domination du Marché")

    st.markdown("""
    <style>
    
    .chart-container{
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        border: 1px solid rgba(255,255,255,0.18);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    /* soft glow overlay */
    .chart-container:before{
        content:"";
        position:absolute; inset:-60px;
        background: radial-gradient(circle at 25% 20%, rgba(255,255,255,0.20), transparent 55%);
        pointer-events:none;
    }

    /* Purple pastel tint for market cap */
    .chart-purple{
        background: linear-gradient(135deg, rgba(196,181,253,0.25), rgba(221,214,254,0.15));
        border-color: rgba(196,181,253,0.40);
    }

    /* Sky blue tint for volume */
    .chart-cyan{
        background: linear-gradient(135deg, rgba(125,211,252,0.22), rgba(186,230,253,0.15));
        border-color: rgba(125,211,252,0.35);
    }

    /* Pastel mauve/pink tint for liquidity */
    .chart-amber{
        background: linear-gradient(135deg, rgba(251,207,232,0.25), rgba(244,194,194,0.15));
        border-color: rgba(251,207,232,0.40);
    }

    .chart-title{
        font-size: 1.1rem;
        font-weight: 900;
        margin: 0 0 0.8rem 0;
        letter-spacing: .3px;
        opacity: .95;
    }
    </style>
    """, unsafe_allow_html=True)

    data = df.dropna(subset=["Market Cap", "Volume"]).copy()
    data = data[data["Market Cap"] > 0]  
    data["LiquidityScore"] = data["Volume"] / data["Market Cap"]

    main = data.sort_values("Market Cap", ascending=False).head(max_coins)

    symbols = main["Symbol"]                
    mcap_share = main["Market Cap"] / main["Market Cap"].sum()
    vol_share = main["Volume"] / main["Volume"].sum()
    liq_score = main["LiquidityScore"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container chart-purple"><p class="chart-title">Part de marché (Market Cap)</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_alpha(0)
        ax.pie(
            mcap_share,
            labels=symbols,
            autopct="%1.1f%%",
            startangle=90,
        )
        centre = plt.Circle((0, 0), 0.70, fc="white", alpha=0.3)
        ax.add_artist(centre)
        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container chart-cyan"><p class="chart-title">Part du flux (Volume 24h)</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_alpha(0)
        ax.pie(
            vol_share,
            labels=symbols,
            autopct="%1.1f%%",
            startangle=90,
        )
        centre = plt.Circle((0, 0), 0.70, fc="white", alpha=0.3)
        ax.add_artist(centre)
        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # Liquidity Score Bar Chart
    st.markdown('<div class="chart-container chart-amber"><p class="chart-title">Score de liquidité</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.bar(symbols, liq_score, alpha=0.85, color='#dda0dd')
    ax.set_ylabel("Volume / Market Cap")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)
