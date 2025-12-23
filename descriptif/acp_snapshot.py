import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ACPSnapshot:
    """
    ACP snapshot sous forme de classe.
    - render(df) : affiche toute la section ACP dans Streamlit
    - garde une structure claire et extensible (state possible si tu veux plus tard)
    """

    def __init__(
        self,
        n_components: int = 3,
        top_n_labels: int = 10,
        ticker_col: str = "Ticker",
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.top_n_labels = top_n_labels
        self.ticker_col = ticker_col
        self.random_state = random_state

    @staticmethod
    def _compute_correlation_circle(X_scaled: np.ndarray, pca: PCA, feature_names: list[str]) -> pd.DataFrame:
        """
        Cercle de corrélation (variables) pour PC1/PC2.
        Coordonnées des variables dans le plan (PC1, PC2) via: loading * sqrt(eigenvalue).
        """
        loadings = pca.components_.T  # (n_features, n_components)
        eigenvalues = pca.explained_variance_  # (n_components,)

        cor = loadings * np.sqrt(eigenvalues)

        cor_df = pd.DataFrame(cor[:, :2], columns=["PC1", "PC2"])
        cor_df["variable"] = feature_names

        cor_df["PC1"] = cor_df["PC1"].clip(-1.1, 1.1)
        cor_df["PC2"] = cor_df["PC2"].clip(-1.1, 1.1)

        return cor_df

    def _prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series]:
        # 1) Sélection AUTOMATIQUE des variables numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = numeric_cols

        if len(selected_cols) == 0:
            raise ValueError("Aucune colonne numérique détectée pour l'ACP.")

        X = df[selected_cols].copy()

        cryptos = (
            df.loc[X.index, self.ticker_col]
            if self.ticker_col in df.columns
            else X.index.astype(str)
        )

        return X, selected_cols, cryptos

    def _fit_pca(self, X: pd.DataFrame) -> tuple[np.ndarray, PCA, np.ndarray, np.ndarray]:
        # 2) Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3) ACP
        n_components = min(self.n_components, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=self.random_state)
        coords = pca.fit_transform(X_scaled)

        explained_var = pca.explained_variance_ratio_

        return X_scaled, pca, coords, explained_var

    def _build_pca_df(self, coords: np.ndarray, cryptos: pd.Series, n_components: int) -> pd.DataFrame:
        pca_df = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(n_components)])
        pca_df["crypto"] = cryptos.values

        # Distance au centre
        if {"PC1", "PC2"}.issubset(pca_df.columns):
            pca_df["distance"] = np.sqrt(pca_df["PC1"] ** 2 + pca_df["PC2"] ** 2)
        else:
            pca_df["distance"] = 0.0

        return pca_df

    def _plot_individuals(self, pca_df: pd.DataFrame) -> None:
        st.markdown("####  Carte des individus (PC1 × PC2)")
        st.caption("Couleur = distance au centre (spécificité). Labels = cryptos les plus atypiques.")

        if not {"PC1", "PC2"}.issubset(pca_df.columns):
            st.warning("Impossible d'afficher la carte PC1/PC2 (pas assez de composantes).")
            return

        fig, ax = plt.subplots(figsize=(9, 6))

        scatter = ax.scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            c=pca_df["distance"],
            cmap="viridis",
            s=120,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.6,
        )

        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

        top_n = min(self.top_n_labels, len(pca_df))
        for _, row in pca_df.sort_values("distance", ascending=False).head(top_n).iterrows():
            ax.text(
                row["PC1"],
                row["PC2"],
                str(row["crypto"]),
                fontsize=9,
                ha="center",
                va="center",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.6),
            )

        ax.set_xlabel("PC1 — Structure globale du marché")
        ax.set_ylabel("PC2 — Profil différenciant")
        ax.set_title("ACP snapshot — cartographie du marché crypto")
        ax.grid(alpha=0.2)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Distance au centre (spécificité)")

        st.pyplot(fig, use_container_width=True)

    def _plot_correlation_circle(self, X: pd.DataFrame, X_scaled: np.ndarray, pca: PCA, selected_cols: list[str]) -> None:
        st.markdown("####  Cercle de corrélation (variables)")
        st.caption("Flèche longue = variable bien représentée ; même direction = corrélation ; opposée = négative.")

        n_components = pca.n_components_

        if X.shape[1] < 2 or n_components < 2:
            st.warning("Pas assez de variables/composantes pour afficher le cercle de corrélation.")
            return

        cor_df = self._compute_correlation_circle(X_scaled, pca, feature_names=selected_cols)

        fig_corr, ax_corr = plt.subplots(figsize=(7, 7))

        theta = np.linspace(0, 2 * np.pi, 400)
        ax_corr.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
        ax_corr.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax_corr.axvline(0, color="grey", linestyle="--", linewidth=0.8)

        for _, row in cor_df.iterrows():
            ax_corr.arrow(
                0, 0,
                row["PC1"], row["PC2"],
                head_width=0.04,
                head_length=0.04,
                length_includes_head=True,
                alpha=0.85
            )
            ax_corr.text(
                row["PC1"] * 1.08,
                row["PC2"] * 1.08,
                str(row["variable"]),
                fontsize=9,
                ha="center",
                va="center"
            )

        ax_corr.set_xlim(-1.1, 1.1)
        ax_corr.set_ylim(-1.1, 1.1)
        ax_corr.set_aspect("equal", adjustable="box")
        ax_corr.set_xlabel("PC1")
        ax_corr.set_ylabel("PC2")
        ax_corr.set_title("Cercle de corrélation — interprétation des axes")
        ax_corr.grid(alpha=0.25)

        st.pyplot(fig_corr, use_container_width=True)

        st.caption(
            "Lecture: flèche longue = variable bien représentée par (PC1, PC2). "
            "Même direction = corrélation positive, opposée = corrélation négative, "
            "perpendiculaire ≈ indépendance."
        )

    def _plot_variance(self, explained_var: np.ndarray) -> None:
        st.markdown("####  Inertie / Variance expliquée")
        st.caption("Barres = variance par composante, courbe = cumul (inertie cumulée).")

        n_components = len(explained_var)
        pcs = [f"PC{i+1}" for i in range(n_components)]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(pcs, explained_var, alpha=0.7)
        ax2.plot(pcs, np.cumsum(explained_var), marker="o", linewidth=2)

        ax2.set_ylabel("Part expliquée")
        ax2.set_title("Variance expliquée (individuelle & cumulée)")
        ax2.grid(axis="y", alpha=0.3)

        for i, v in enumerate(np.cumsum(explained_var)):
            ax2.text(i, v + 0.02, f"{v:.0%}", ha="center")

        st.pyplot(fig2, use_container_width=True)

    def _render_interpretation(self, explained_var: np.ndarray) -> None:
        st.markdown("###  Interprétation ")
        col_var, col_text = st.columns([1.0, 1.2], gap="large")

        with col_var:
            with st.container(border=True):
                self._plot_variance(explained_var)

        with col_text:
            with st.container(border=True):
                st.markdown("####  Objectif de l’ACP")
                st.caption("Résumer le marché crypto dans un plan 2D pour révéler structure,"
                           "similarités et profils atypiques."
                           )

                st.markdown("####  Lecture des axes")

                pc1_pct = explained_var[0] if len(explained_var) > 0 else 0
                pc2_pct = explained_var[1] if len(explained_var) > 1 else 0

                st.markdown(
                    f"""
- **PC1 ({pc1_pct:.0%}) – Facteur dominant du marché**  
➜ Résume la **structure globale** (taille / activité / intensité) des cryptomonnaies.

- **PC2 ({pc2_pct:.0%}) – Facteur de différenciation**  
➜ Sépare des profils **plus stables** de profils **plus atypiques / risqués**.
"""
                )

    def render(self, df: pd.DataFrame) -> None:
        """
        Point d'entrée Streamlit: affiche toute la section ACP snapshot.
        """
        st.divider()

        try:
            X, selected_cols, cryptos = self._prepare_data(df)
            X_scaled, pca, coords, explained_var = self._fit_pca(X)
        except Exception as e:
            st.error(f"Erreur ACP: {e}")
            return

        n_components = pca.n_components_
        pca_df = self._build_pca_df(coords, cryptos, n_components=n_components)

        col_ind, col_cor = st.columns([1.2, 1.0], gap="large")

        # INDIVIDUS
        with col_ind:
            with st.container(border=True):
                self._plot_individuals(pca_df)

        # CERCLE DE CORRÉLATION
        with col_cor:
            with st.container(border=True):
                self._plot_correlation_circle(X, X_scaled, pca, selected_cols)

        st.divider()
        self._render_interpretation(explained_var)

        st.divider()
        st.success(
            "Cette ACP fournit une cartographie claire du marché crypto et un cercle de corrélation "
            "pour interpréter les axes. Les cryptos proches ont des caractéristiques similaires, "
            "celles éloignées du centre ont un profil plus spécifique."
        )
