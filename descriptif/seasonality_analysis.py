import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


class SeasonalityAnalysis:
    """
    Analyse saisonnière (BTC-USD par défaut) sur Sep-Nov:
    - Boxplot des rendements par jour de semaine
    - Meilleur jour moyen (metric)
    """

    def __init__(
        self,
        ticker: str = "BTC-USD",
        months: list[int] | None = None,
        figsize: tuple[float, float] = (6, 3),
    ) -> None:
        self.ticker = ticker
        self.months = months if months is not None else [9, 10, 11]
        self.figsize = figsize

    def render(self, df) -> None:
        df_filtered = df[(df["Date"].dt.month.isin(self.months)) & (df["Ticker"] == self.ticker)]

        if df_filtered.empty:
            st.warning(f"Aucune donnée disponible pour {self.ticker} sur la période Sep-Nov")
            return

        df_ticker = df_filtered.sort_values("Date").copy()
        df_ticker["Returns"] = df_ticker["Close"].pct_change()
        df_ticker["Day_of_Week"] = df_ticker["Date"].dt.day_name()

        weekly_stats = df_ticker.groupby("Day_of_Week")["Returns"].agg(["mean", "std", "count"])
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_stats = weekly_stats.reindex(order)

        # Boxplot
        fig, ax = plt.subplots(figsize=self.figsize)
        data_by_day = [df_ticker[df_ticker["Day_of_Week"] == day]["Returns"] * 100 for day in order]
        box = ax.boxplot(data_by_day, labels=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"], patch_artist=True)

        colors_box = ["#667eea", "#7c6eea", "#9265db", "#a55dcb", "#b856bc", "#c950ad", "#d94b9e"]
        for patch, color in zip(box["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("#2d3748")
            patch.set_linewidth(1.5)
        for whisker in box["whiskers"]:
            whisker.set_color("#4a5568")
            whisker.set_linewidth(1.2)
        for cap in box["caps"]:
            cap.set_color("#4a5568")
            cap.set_linewidth(1.2)
        for median in box["medians"]:
            median.set_color("#ffffff")
            median.set_linewidth(2)
        for flier in box["fliers"]:
            flier.set_markerfacecolor("#e53e3e")
            flier.set_markeredgecolor("#e53e3e")
            flier.set_markersize(4)

        ax.axhline(0, color="#718096", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel("Rendements (%)", fontsize=8, color="#4a5568")
        ax.set_xlabel("", fontsize=8)
        ax.set_title(f"Distribution des rendements - {self.ticker}", fontsize=9, fontweight="bold", color="#2d3748")
        ax.tick_params(labelsize=7, colors="#4a5568")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e2e8f0")
        ax.spines["bottom"].set_color("#e2e8f0")
        ax.grid(True, alpha=0.4, axis="y", linestyle="--", color="#e2e8f0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Best day metric
        if weekly_stats["mean"].dropna().empty:
            st.info("Pas assez de données pour calculer le meilleur jour.")
            return

        best_day = weekly_stats["mean"].idxmax()
        best_return = weekly_stats["mean"].max() * 100

        # Petit affichage propre
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Meilleur jour (moyenne)", best_day)
        with col2:
            st.metric("Rendement moyen", f"{best_return:.2f} %")
