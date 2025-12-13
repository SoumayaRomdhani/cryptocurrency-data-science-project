import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Helpers data

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _select_features(data: pd.DataFrame) -> list[str]:
    candidates = ["Market Cap", "Volume", "Price", "Change %", "52 WkChange %"]
    return [c for c in candidates if c in data.columns]


def _prepare_X(data: pd.DataFrame, features: list[str], use_log: bool) -> pd.DataFrame:
    X = data[features].copy()
    for c in features:
        X[c] = _to_num(X[c])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if use_log:
        for c in ["Market Cap", "Volume"]:
            if c in X.columns:
                X[c] = np.log1p(np.clip(X[c], a_min=0, a_max=None))
    return X


def _auto_pca_components(
    X_scaled: np.ndarray,
    target: float = 0.90,
    min_keep: float = 0.80,
    max_keep: float = 0.95
):
    # PCA
    pca_full = PCA(n_components=min(X_scaled.shape), random_state=42)
    pca_full.fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)

    target = float(np.clip(target, min_keep, max_keep))
    n = int(np.searchsorted(cum, target) + 1)

    if min(X_scaled.shape) >= 2:
        n = max(2, n)
    n = min(n, min(X_scaled.shape))
    return n, pca_full, cum, target


def _auto_best_k(Z: np.ndarray, max_k: int = 6):
    n_samples = Z.shape[0]
    k_min = 2
    k_max = min(max_k, n_samples - 1)
    if k_max < 2:
        return None, [], [], []

    ks = list(range(k_min, k_max + 1))
    sils, inertias = [], []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Z)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Z, labels))

    best_idx = int(np.argmax(sils))
    return ks[best_idx], ks, sils, inertias


def _fmt_big(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    x = float(x)
    if abs(x) >= 1e12:
        return f"{x/1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"{x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.4g}"


def display_kmeans_clustering(df_snapshot: pd.DataFrame, max_k: int = 6) -> None:

    st.title(" Crypto Clustering Lab")
    st.caption("Pipeline : StandardScaler → PCA (auto 80–95%) → K-means (K auto via silhouette)")

    data = df_snapshot.copy()
    features = _select_features(data)
    use_log = True
    target_var = 0.90

    if len(features) < 2:
        st.warning("Pas assez de colonnes numériques pour faire PCA + K-means.")
        return

    #  Prepare
    X = _prepare_X(data, features, use_log=use_log)
    if len(X) < 3:
        st.warning("Il faut au moins 3 cryptos pour un clustering robuste.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #  PCA auto
    n_components, pca_full, cum, target_used = _auto_pca_components(
        X_scaled, target=target_var, min_keep=0.80, max_keep=0.95
    )
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cum_reduced = np.cumsum(explained)

    # K auto
    best_k, ks, sils, inertias = _auto_best_k(Z, max_k=max_k)
    if best_k is None:
        st.warning("Pas assez de points pour choisir K automatiquement.")
        return

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Z)
    sil_final = silhouette_score(Z, labels)

    out = data.copy()
    out["cluster"] = labels

    # Premium KPI cards
    with st.container(border=True):
        st.subheader(" Résumé rapide")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cryptos", f"{len(out)}")
        c2.metric("Features", f"{len(features)}")
        c3.metric("PCs retenues", f"{n_components}", help=f"Cible ≈ {target_used:.0%} de variance")
        c4.metric("K optimal", f"{best_k}", help=f"Silhouette = {sil_final:.3f}")

        st.caption(f"Variance expliquée par les PCs retenues : **{cum_reduced[-1]:.1%}**")

    #  Tabs
    tab_overview, tab_clusters, tab_profiles = st.tabs(
        [" Overview", " Clusters", " Profils & Insights"]
    )

    # Tab 1: Overview (PCA & K)

    with tab_overview:
        st.markdown("##  Overview — PCA & choix de K")

        row1 = st.columns([1.35, 1.0])
        with row1[0]:
            with st.container(border=True):
                st.markdown("###  PCA — Variance expliquée (pro)")

                # 1) Courbe cumulée + annotations
                fig, ax = plt.subplots(figsize=(8, 4))
                pcs = np.arange(1, len(pca_full.explained_variance_ratio_) + 1)

                ax.step(pcs, cum, where="mid")
                ax.scatter(pcs, cum, s=25)

                ax.axhline(target_used, linestyle="--", linewidth=1)
                ax.axvline(n_components, linestyle="--", linewidth=1)

                ax.set_xlabel("Nombre de composantes (PCs)")
                ax.set_ylabel("Variance cumulée expliquée")
                ax.set_title("PCA — Cumul de variance (avec seuil & PCs retenues)")
                ax.grid(alpha=0.25)

                ax.text(
                    n_components, min(0.98, cum[n_components - 1] + 0.03),
                    f"PCs retenues = {n_components}",
                    fontsize=9
                )
                ax.text(
                    pcs[-1], target_used,
                    f"cible {target_used:.0%}",
                    fontsize=9, ha="right", va="bottom"
                )
                st.pyplot(fig, use_container_width=True)

                # 2) Bar chart variance par PC
                fig, ax = plt.subplots(figsize=(8, 3.5))
                evr = pca_full.explained_variance_ratio_
                top = min(len(evr), 8)
                ax.bar(np.arange(1, top + 1), evr[:top])
                ax.set_xlabel("PC")
                ax.set_ylabel("Variance expliquée")
                ax.set_title("Variance expliquée par PC (top)")
                ax.grid(axis="y", alpha=0.25)
                st.pyplot(fig, use_container_width=True)

        with row1[1]:
            with st.container(border=True):
                st.markdown("###  K optimal — lecture rapide")

                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.plot(ks, sils, marker="o")
                ax.axvline(best_k, linestyle="--", linewidth=1)
                ax.set_xlabel("K")
                ax.set_ylabel("Silhouette")
                ax.set_title("Silhouette score (max = meilleur K)")
                ax.grid(alpha=0.25)

                # annotation best k
                best_sil = max(sils) if len(sils) else None
                if best_sil is not None:
                    ax.text(best_k, best_sil, f"best K={best_k}", fontsize=9, ha="left", va="bottom")
                st.pyplot(fig, use_container_width=True)

                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.plot(ks, inertias, marker="o")
                ax.axvline(best_k, linestyle="--", linewidth=1)
                ax.set_xlabel("K")
                ax.set_ylabel("Inertia")
                ax.set_title("Elbow (inertia)")
                ax.grid(alpha=0.25)
                st.pyplot(fig, use_container_width=True)

    # Tab 2: Clusters

    with tab_clusters:
        with st.container(border=True):
            st.markdown("###  Carte des cryptos (PCA PC1/PC2 + clusters)")
            if Z.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(9, 6))
                sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, alpha=0.9)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("Projection PCA (PC1/PC2) + clusters K-means")
                ax.grid(alpha=0.25)

                for i in range(len(out)):
                    name = out["Symbol"].iloc[i] if "Symbol" in out.columns else str(i)
                    ax.text(Z[i, 0], Z[i, 1], str(name), fontsize=8)

                handles, _ = sc.legend_elements()
                ax.legend(handles, [f"Cluster {i}" for i in range(best_k)], title="Clusters", loc="best")
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Pas de PC2 disponible (trop peu de dimensions).")

        # Focus cluster
        with st.container(border=True):
            st.markdown("###  Focus cluster")
            cluster_ids = sorted(out["cluster"].unique().tolist())
            selected = st.selectbox("Choisir un cluster", cluster_ids)

            # Top cryptos dans le cluster (par Market Cap si dispo)
            st.markdown("**Cryptos dans ce cluster :**")
            cols_id = [c for c in ["Symbol", "Name"] if c in out.columns]
            sub = out[out["cluster"] == selected].copy()

            if "Market Cap" in sub.columns:
                sub["Market Cap"] = pd.to_numeric(sub["Market Cap"], errors="coerce")
                sub = sub.sort_values("Market Cap", ascending=False)

            st.dataframe(sub[cols_id + ["cluster"]], use_container_width=True, hide_index=True)

    # Tab 3: Profiles & Insights

    with tab_profiles:
        st.markdown("##  Profils & Insights")

        prof = out.groupby("cluster")[features].mean(numeric_only=True)
        prof_z = (prof - prof.mean(axis=0)) / (prof.std(axis=0) + 1e-9)

        # Heatmap premium
        with st.container(border=True):
            st.markdown("###  Heatmap  — profil relatif (z-score)")

            #  Heatmap graphique annotée
            fig, ax = plt.subplots(figsize=(10, 3.8))
            vmax = np.nanmax(np.abs(prof_z.values))
            im = ax.imshow(prof_z.values, aspect="auto", vmin=-vmax, vmax=vmax, cmap="PuOr")

            ax.set_yticks(range(len(prof_z.index)))
            ax.set_yticklabels([f"Cluster {i}" for i in prof_z.index])

            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=25, ha="right")

            ax.set_title("Heatmap z-score (annotée)")
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

            for i in range(prof_z.shape[0]):
                for j in range(prof_z.shape[1]):
                    ax.text(j, i, f"{prof_z.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

            st.pyplot(fig, use_container_width=True)
