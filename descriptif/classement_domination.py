import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


class ClassementDomination:
    """
    Classe Streamlit pour afficher:
    - Part de marché (Market Cap) (donut)
    - Part du flux (Volume 24h) (donut)
    - Score de liquidité (bar chart)
    """

    def __init__(self, default_max_coins: int = 9) -> None:
        self.default_max_coins = default_max_coins

    @staticmethod
    def _inject_css() -> None:
        st.markdown(
            """
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

            .chart-container:before{
                content:"";
                position:absolute; inset:-60px;
                background: radial-gradient(circle at 25% 20%, rgba(255,255,255,0.20), transparent 55%);
                pointer-events:none;
            }

            .chart-purple{
                background: linear-gradient(135deg, rgba(196,181,253,0.25), rgba(221,214,254,0.15));
                border-color: rgba(196,181,253,0.40);
            }

            .chart-cyan{
                background: linear-gradient(135deg, rgba(125,211,252,0.22), rgba(186,230,253,0.15));
                border-color: rgba(125,211,252,0.35);
            }

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
            """,
            unsafe_allow_html=True
        )

    @staticmethod
    def _compute_tables(df: pd.DataFrame, max_coins: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        data = df.dropna(subset=["Market Cap", "Volume"]).copy()
        data = data[data["Market Cap"] > 0]
        data["LiquidityScore"] = data["Volume"] / data["Market Cap"]

        main = data.sort_values("Market Cap", ascending=False).head(max_coins)

        symbols = main["Symbol"]
        mcap_share = main["Market Cap"] / main["Market Cap"].sum()
        vol_share = main["Volume"] / main["Volume"].sum()
        liq_score = main["LiquidityScore"]

        return symbols, mcap_share, vol_share, liq_score

    @staticmethod
    def _plot_donut(labels: pd.Series, values: pd.Series, title: str, css_class: str) -> None:
        st.markdown(
            f'<div class="chart-container {css_class}"><p class="chart-title">{title}</p>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_alpha(0)
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        centre = plt.Circle((0, 0), 0.70, fc="white", alpha=0.3)
        ax.add_artist(centre)
        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    @staticmethod
    def _plot_liquidity(labels: pd.Series, liq_score: pd.Series) -> None:
        st.markdown(
            '<div class="chart-container chart-amber"><p class="chart-title">Score de liquidité</p>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(4.2, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.bar(labels, liq_score, alpha=0.85, color="#dda0dd")
        ax.set_ylabel("Vol / MCap", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    def render(self, df: pd.DataFrame, max_coins: int | None = None) -> None:
        st.subheader(" Classement & Domination du Marché")
        self._inject_css()

        max_coins = self.default_max_coins if max_coins is None else max_coins

        try:
            symbols, mcap_share, vol_share, liq_score = self._compute_tables(df, max_coins=max_coins)
        except Exception as e:
            st.error(f"Erreur Classement/Domination: {e}")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            self._plot_donut(symbols, mcap_share, "Part de marché (Market Cap)", "chart-purple")

        with col2:
            self._plot_donut(symbols, vol_share, "Part du flux (Volume 24h)", "chart-cyan")

        with col3:
            self._plot_liquidity(symbols, liq_score)
