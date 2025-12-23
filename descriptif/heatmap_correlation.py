import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


class CorrelationHeatmap:
    """
    Heatmap de corrélation des prix (Close) sur Sep-Nov (par défaut).
    """

    def __init__(self, months: list[int] | None = None, figsize: tuple[int, int] = (5, 4)) -> None:
        self.months = months if months is not None else [9, 10, 11]
        self.figsize = figsize

    def render(self, df) -> None:
        df_filtered = df[df["Date"].dt.month.isin(self.months)]
        corr_matrix = df_filtered.pivot_table(index="Date", columns="Ticker", values="Close")
        corr = corr_matrix.corr()

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 7},
            ax=ax,
        )
        ax.set_title("Corrélation des Prix (Sep-Nov)", fontsize=9, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
