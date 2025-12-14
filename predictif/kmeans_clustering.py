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


# Styles

def _inject_glass_kpi_css() -> None:
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

  /* black icon */
  color: rgba(0,0,0,0.82) !important;
  filter: drop-shadow(0 8px 16px rgba(0,0,0,0.18)) !important;
}

/* ---------- Selectbox compact (less wide) ---------- */
div[data-testid="stSelectbox"]{
  max-width: 640px;
}
div[data-testid="stSelectbox"] [data-baseweb="select"]{
  max-width: 640px;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  width: 100% !important;
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
  box-shadow: 0 0 0 5px rgba(99,102,241,0.17), 0 16px 40px rgba(0,0,0,0.24) !important;
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

/* ---------- KPI cards ---------- */
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
.kpi-amber{
  background: linear-gradient(135deg, rgba(245,158,11,0.16), rgba(239,68,68,0.10));
  border-color: rgba(245,158,11,0.28);
}
.kpi-slate{
  background: linear-gradient(135deg, rgba(148,163,184,0.16), rgba(59,130,246,0.08));
  border-color: rgba(148,163,184,0.26);
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
  font-size: 2.15rem;
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
.chip-amber{
  background: linear-gradient(135deg, rgba(245,158,11,0.44), rgba(239,68,68,0.28));
  box-shadow: 0 10px 26px rgba(245,158,11,0.18);
}
.chip-slate{
  background: linear-gradient(135deg, rgba(148,163,184,0.40), rgba(59,130,246,0.22));
  box-shadow: 0 10px 26px rgba(148,163,184,0.18);
}

/* ---------- Glass table wrapper ---------- */
.glass-table{
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.16);
  background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow: 0 18px 45px rgba(0,0,0,0.18);
  padding: 14px 14px 6px 14px;
  overflow: hidden;
}
</style>
        """,
        unsafe_allow_html=True
    )


def _cluster_palette():
    return [
        {"dot": "#7C3AED", "row_css": "rgba(124,58,237,0.10)", "row_mpl": (124 / 255, 58 / 255, 237 / 255, 0.16)},
        {"dot": "#06B6D4", "row_css": "rgba(6,182,212,0.10)", "row_mpl": (6 / 255, 182 / 255, 212 / 255, 0.16)},
        {"dot": "#F59E0B", "row_css": "rgba(245,158,11,0.10)", "row_mpl": (245 / 255, 158 / 255, 11 / 255, 0.18)},
        {"dot": "#EF4444", "row_css": "rgba(239,68,68,0.10)", "row_mpl": (239 / 255, 68 / 255, 68 / 255, 0.16)},
        {"dot": "#22C55E", "row_css": "rgba(34,197,94,0.10)", "row_mpl": (34 / 255, 197 / 255, 94 / 255, 0.16)},
        {"dot": "#3B82F6", "row_css": "rgba(59,130,246,0.10)", "row_mpl": (59 / 255, 130 / 255, 246 / 255, 0.16)},
        {"dot": "#E11D48", "row_css": "rgba(225,29,72,0.10)", "row_mpl": (225 / 255, 29 / 255, 72 / 255, 0.16)},
        {"dot": "#A855F7", "row_css": "rgba(168,85,247,0.10)", "row_mpl": (168 / 255, 85 / 255, 247 / 255, 0.16)},
    ]


def _render_glass_cluster_table(df: pd.DataFrame, cluster_col: str = "cluster") -> None:
    pal = _cluster_palette()

    df_show = df.copy()
    if cluster_col in df_show.columns:
        df_show[cluster_col] = df_show[cluster_col].astype(int)

    def _row_style(row):
        if cluster_col not in row.index:
            return [""] * len(row)
        c = int(row[cluster_col])
        tone = pal[c % len(pal)]["row_css"]
        return [f"background: {tone}; border-bottom: 1px solid rgba(255,255,255,0.10);" for _ in row]

    sty = (
        df_show.style
        .apply(_row_style, axis=1)
        .set_table_styles([
            {"selector": "table", "props": [
                ("width", "100%"),
                ("border-collapse", "separate"),
                ("border-spacing", "0 10px"),
            ]},
            {"selector": "thead th", "props": [
                ("text-align", "left"),
                ("font-weight", "900"),
                ("letter-spacing", ".2px"),
                ("border", "none"),
                ("padding", "10px 12px"),
                ("background", "rgba(255,255,255,0.08)"),
                ("backdrop-filter", "blur(12px)"),
                ("-webkit-backdrop-filter", "blur(12px)"),
                ("border-radius", "12px"),
            ]},
            {"selector": "tbody td", "props": [
                ("border", "none"),
                ("padding", "12px 12px"),
                ("font-weight", "650"),
                ("border-radius", "12px"),
            ]},
            {"selector": "tbody tr:hover td", "props": [
                ("filter", "brightness(1.06)"),
                ("transform", "translateY(-1px)"),
                ("transition", "all .12s ease"),
            ]},
        ])
    )

    st.markdown('<div class="glass-table">', unsafe_allow_html=True)
    st.markdown(sty.to_html(), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Main

def display_kmeans_clustering(df_snapshot: pd.DataFrame, max_k: int = 6) -> None:
    _inject_glass_kpi_css()
    data = df_snapshot.copy()
    features = _select_features(data)
    use_log = True
    target_var = 0.90

    if len(features) < 2:
        st.warning("Pas assez de colonnes numériques pour faire PCA + K-means.")
        return

    X = _prepare_X(data, features, use_log=use_log)
    if len(X) < 3:
        st.warning("Il faut au moins 3 cryptos pour un clustering robuste.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components, pca_full, cum, target_used = _auto_pca_components(
        X_scaled, target=target_var, min_keep=0.80, max_keep=0.95
    )
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cum_reduced = np.cumsum(explained)

    best_k, ks, sils, inertias = _auto_best_k(Z, max_k=max_k)
    if best_k is None:
        st.warning("Pas assez de points pour choisir K automatiquement.")
        return

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Z)
    sil_final = silhouette_score(Z, labels)

    out = data.copy()
    out["cluster"] = labels

    #  KPI cards
    st.subheader(" Résumé rapide")

    c1, c3, c4 = st.columns(3, gap="large")

    with c1:
        st.markdown(
            f"""
<div class="kpi-card kpi-purple">
  <div class="kpi-head">
    <p class="kpi-title">Cryptos</p>
    <div class="icon-chip chip-purple"><span class="ms">token</span></div>
  </div>
  <p class="kpi-value">{len(out)}</p>
  <p class="kpi-caption">Taille de l’univers analysé.</p>
</div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
<div class="kpi-card kpi-slate">
  <div class="kpi-head">
    <p class="kpi-title">PCs retenues</p>
    <div class="icon-chip chip-slate"><span class="ms">scatter_plot</span></div>
  </div>
  <p class="kpi-value">{n_components}</p>
  <p class="kpi-caption">Cible variance ≈ {target_used:.0%}</p>
</div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
<div class="kpi-card kpi-amber">
  <div class="kpi-head">
    <p class="kpi-title">K optimal</p>
    <div class="icon-chip chip-amber"><span class="ms">hub</span></div>
  </div>
  <p class="kpi-value">{best_k}</p>
  <p class="kpi-caption">Silhouette = {sil_final:.3f}</p>
</div>
            """,
            unsafe_allow_html=True
        )

    st.caption(f"Variance expliquée par les PCs retenues : **{cum_reduced[-1]:.1%}**")

    tab_overview, tab_clusters, tab_profiles = st.tabs(
        [" Overview", " Clusters", " Profils & Insights"]
    )

    # Tab 1: Overview

    with tab_overview:
        st.markdown("##  Overview — PCA & choix de K")

        row1 = st.columns([1.35, 1.0])
        with row1[0]:
            with st.container(border=True):
                st.markdown("###  PCA — Variance expliquée (pro)")

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
        pal = _cluster_palette()

        with st.container(border=True):
            st.markdown("###  Carte des cryptos (PCA PC1/PC2 + clusters)")

            if Z.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(9, 6))

                point_colors = [pal[int(c) % len(pal)]["dot"] for c in labels]

                # glow ring behind points
                ax.scatter(
                    Z[:, 0], Z[:, 1],
                    s=360,
                    c=point_colors,
                    alpha=0.20,
                    linewidths=0
                )

                # colored points
                ax.scatter(
                    Z[:, 0], Z[:, 1],
                    s=95,
                    c=point_colors,
                    alpha=0.95,
                    edgecolors="white",
                    linewidths=1.0
                )

                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("Projection PCA (PC1/PC2) + clusters K-means")
                ax.grid(alpha=0.22)

                for i in range(len(out)):
                    name = out["Symbol"].iloc[i] if "Symbol" in out.columns else str(i)
                    c = int(labels[i])
                    tint_mpl = pal[c % len(pal)]["row_mpl"]
                    ax.text(
                        Z[i, 0], Z[i, 1],
                        str(name),
                        fontsize=8,
                        ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.25", fc=tint_mpl, ec="none")
                    )

                from matplotlib.lines import Line2D
                handles = []
                for c in range(best_k):
                    col = pal[c % len(pal)]["dot"]
                    handles.append(
                        Line2D([0], [0], marker="o", color="w", label=f"Cluster {c}",
                               markerfacecolor=col, markeredgecolor="white", markersize=9)
                    )
                ax.legend(handles=handles, title="Clusters", loc="best")

                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Pas de PC2 disponible (trop peu de dimensions).")

        with st.container(border=True):
            st.markdown("###  Focus cluster")

            cluster_ids = sorted(out["cluster"].unique().tolist())
            selected = st.selectbox("Choisir un cluster", cluster_ids)

            st.markdown("**Cryptos dans ce cluster :**")

            cols_id = [c for c in ["Symbol", "Name"] if c in out.columns]
            sub = out[out["cluster"] == selected].copy()

            if "Market Cap" in sub.columns:
                sub["Market Cap"] = pd.to_numeric(sub["Market Cap"], errors="coerce")
                sub = sub.sort_values("Market Cap", ascending=False)

            extra = []
            for c in ["Market Cap", "Volume", "Price", "Change %"]:
                if c in sub.columns:
                    extra.append(c)

            sub_show = sub[cols_id + extra + ["cluster"]].copy()

            for c in ["Market Cap", "Volume"]:
                if c in sub_show.columns:
                    sub_show[c] = pd.to_numeric(sub_show[c], errors="coerce").apply(_fmt_big)

            _render_glass_cluster_table(sub_show, cluster_col="cluster")

    # Tab 3: Profiles & Insights

    with tab_profiles:
        st.markdown("##  Profils & Insights")

        prof = out.groupby("cluster")[features].mean(numeric_only=True)
        prof_z = (prof - prof.mean(axis=0)) / (prof.std(axis=0) + 1e-9)

        #  Heatmap
        with st.container(border=True):
            st.markdown("###  Heatmap  — profil relatif (z-score)")

            import matplotlib.patheffects as pe

            fig, ax = plt.subplots(figsize=(10.5, 4.2))

            v = prof_z.values.astype(float)
            vmax = float(np.nanpercentile(np.abs(v), 95)) if np.isfinite(v).any() else 1.0
            vmax = max(vmax, 1e-6)

            cmap = plt.get_cmap("RdBu_r")
            im = ax.imshow(v, aspect="auto", vmin=-vmax, vmax=vmax, cmap=cmap, interpolation="nearest")

            # Ticks
            ax.set_yticks(range(len(prof_z.index)))
            ax.set_yticklabels([f"Cluster {i}" for i in prof_z.index], fontweight="bold")

            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=25, ha="right", fontweight="bold")

            ax.set_title("Heatmap z-score (annotée)", fontweight="bold", pad=10)

            # Gridlines between cells (subtle but makes reading easier)
            ax.set_xticks(np.arange(-.5, v.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, v.shape[0], 1), minor=True)
            ax.grid(which="minor", linestyle="-", linewidth=0.8, alpha=0.18)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Colorbar
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=9)

            # Annotations
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    val = v[i, j]
                    if not np.isfinite(val):
                        txt = "NA"
                        tcolor = "black"
                    else:
                        txt = f"{val:.2f}"
                        # normalize to [0,1] and estimate perceived brightness
                        norm = (val + vmax) / (2 * vmax)
                        r, g, b, _ = cmap(norm)
                        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                        tcolor = "black" if luminance > 0.62 else "white"

                    ax.text(
                        j, i, txt,
                        ha="center", va="center",
                        fontsize=10,
                        fontweight="bold",
                        color=tcolor,
                        path_effects=[
                            pe.Stroke(linewidth=2.6, foreground="black" if tcolor == "white" else "white", alpha=0.55),
                            pe.Normal()
                        ]
                    )

            # Make layout tight so labels don’t clip
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
