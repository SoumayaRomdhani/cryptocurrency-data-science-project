import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


class PriceEvolution:
    """
    Évolution des prix (Close) par ticker, sur Sep-Nov par défaut.
    """

    def __init__(
        self,
        months: list[int] | None = None,
        figsize: tuple[float, float] = (5, 2.5),
        colors: list[str] | None = None,
    ) -> None:
        self.months = months if months is not None else [9, 10, 11]
        self.figsize = figsize
        self.colors = colors if colors is not None else ["#667eea", "#764ba2", "#48bb78", "#ed8936", "#e53e3e"]

    def render(self, df) -> None:
        df_filtered = df[df["Date"].dt.month.isin(self.months)]

        tickers = df_filtered["Ticker"].unique()
        cols = st.columns(2)

        for idx, ticker in enumerate(tickers):
            df_ticker = df_filtered[df_filtered["Ticker"] == ticker].sort_values("Date")
            color = self.colors[idx % len(self.colors)]

            fig, ax = plt.subplots(figsize=self.figsize)
            ax.fill_between(df_ticker["Date"], df_ticker["Close"], alpha=0.15, color=color)
            ax.plot(df_ticker["Date"], df_ticker["Close"], color=color, linewidth=2)

            ax.set_title(f"{ticker}", fontsize=9, fontweight="bold", color="#2d3748")
            ax.set_xlabel("", fontsize=7)
            ax.set_ylabel("USD", fontsize=7, color="#4a5568")
            ax.tick_params(axis="x", rotation=45, labelsize=6, colors="#4a5568")
            ax.tick_params(axis="y", labelsize=6, colors="#4a5568")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#e2e8f0")
            ax.spines["bottom"].set_color("#e2e8f0")
            ax.grid(True, alpha=0.4, linestyle="--", color="#e2e8f0")
            plt.tight_layout()

            with cols[idx % 2]:
                st.pyplot(fig)

            plt.close(fig)
