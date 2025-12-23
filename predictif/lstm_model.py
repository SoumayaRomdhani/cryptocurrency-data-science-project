import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#48bb78',
    'warning': '#ed8936',
    'danger': '#e53e3e',
    'text': '#2d3748'
}


class ModelResultsViewer:
    """
    Viewer des résultats pré-calculés (pickle):
    - métriques train/test (RMSE, MAE, MAPE, R²)
    - prix futur (LSTM)
    - courbes y_true vs y_pred train/test
    """

    def __init__(self, default_pkl_path: str = "data/model_results.pkl") -> None:
        self.default_pkl_path = default_pkl_path

    @staticmethod
    def _inject_css() -> None:
        st.markdown("""
        <style>
        /* ===================== METRIC SECTION CARDS ===================== */
        .metric-section{
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
        .metric-section:before{
            content:"";
            position:absolute; inset:-60px;
            background: radial-gradient(circle at 25% 20%, rgba(255,255,255,0.20), transparent 55%);
            pointer-events:none;
        }

        /* Test section - purple/indigo tint */
        .metric-test{
            background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(168,85,247,0.10));
            border-color: rgba(99,102,241,0.30);
        }

        /* Train section - cyan/green tint */
        .metric-train{
            background: linear-gradient(135deg, rgba(34,211,238,0.16), rgba(72,187,120,0.10));
            border-color: rgba(34,211,238,0.28);
        }

        .metric-title{
            font-size: 1.1rem;
            font-weight: 900;
            margin: 0 0 1rem 0;
            letter-spacing: .3px;
            opacity: .95;
        }

        .metric-grid{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .metric-item{
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }

        .metric-label{
            font-size: 0.8rem;
            font-weight: 600;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-value{
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def _load_precomputed_results(pkl_path: str) -> dict:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            st.error(f"Fichier introuvable : {pkl_path}")
            return {}
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return {}

    @staticmethod
    def _fmt(x, kind: str = "float") -> str:
        if x is None:
            return "NA"
        try:
            if kind == "pct":
                return "{:.2f}%".format(float(x))
            if kind == "r2":
                return "{:.4f}".format(float(x))
            return "{:.2f}".format(float(x))
        except Exception:
            return "NA"

    def render(self, pkl_path: str | None = None, model_filter: str | None = None) -> None:
        """
        Remplace l'ancienne fonction display_model_results(pkl_path=..., model_filter=...)
        """
        pkl_path = self.default_pkl_path if pkl_path is None else pkl_path

        data = self._load_precomputed_results(pkl_path)
        if not data:
            return

        self._inject_css()

        model_names = {'lstm': 'LSTM', 'xgb': 'XGBoost'}

        if model_filter:
            models = [model_filter] if model_filter in data else []
        else:
            models = ['lstm', 'xgb']

        for model_key in models:
            if model_key not in data:
                st.warning(f"Résultats manquants pour {model_names.get(model_key, model_key)}")
                continue

            res = data[model_key]
            col_test, col_train = st.columns(2)

            # --- TEST METRICS
            with col_test:
                rmse_test_str = self._fmt(res.get("rmse_test"))
                mae_test_str = self._fmt(res.get("mae_test"))
                mape_test_str = self._fmt(res.get("mape_test"), kind="pct")
                r2_test_str = self._fmt(res.get("r2_test"), kind="r2")

                st.markdown(
                    f"""
                    <div class="metric-section metric-test">
                        <p class="metric-title">Métriques Test</p>
                        <div class="metric-grid">
                            <div class="metric-item">
                                <span class="metric-label">RMSE</span>
                                <span class="metric-value">{rmse_test_str}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">MAE</span>
                                <span class="metric-value">{mae_test_str}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">MAPE</span>
                                <span class="metric-value">{mape_test_str}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">R²</span>
                                <span class="metric-value">{r2_test_str}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # --- TRAIN METRICS
            with col_train:
                rmse_train_str = self._fmt(res.get("rmse_train"))
                mae_train_str = self._fmt(res.get("mae_train"))
                mape_train_str = self._fmt(res.get("mape_train"), kind="pct")
                r2_train_str = self._fmt(res.get("r2_train"), kind="r2")

                st.markdown(
                    f"""
                    <div class="metric-section metric-train">
                        <p class="metric-title">Métriques Train</p>
                        <div class="metric-grid">
                            <div class="metric-item">
                                <span class="metric-label">RMSE</span>
                                <span class="metric-value">{rmse_train_str}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">MAE</span>
                                <span class="metric-value">{mae_train_str}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">MAPE</span>
                                <span class="metric-value">{mape_train_str}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-label">R²</span>
                                <span class="metric-value">{r2_train_str}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # --- Future price for LSTM
            if model_key == 'lstm':
                future_price = res.get("future_price")
                last_price = res.get("last_known_price")
                if future_price is not None and last_price is not None:
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Dernier prix connu", f"{float(last_price):.2f} USD")
                    with col2:
                        delta = float(future_price) - float(last_price)
                        st.metric("Prix prédit (demain)", f"{float(future_price):.2f} USD", delta=f"{delta:+.2f} USD")

            # --- Curves
            y_true_test = res.get("y_test_true")
            y_pred_test = res.get("y_test_pred")
            y_true_train = res.get("y_train_true")
            y_pred_train = res.get("y_train_pred")

            if y_true_test is not None and y_pred_test is not None:
                st.markdown("---")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

                # Test
                ax1.plot(y_true_test, label="Réel", color=COLORS['primary'], linewidth=1.5)
                ax1.plot(y_pred_test, label="Prédit", color=COLORS['danger'], linewidth=1.5, alpha=0.8)
                ax1.set_title("Test", fontsize=10, fontweight='bold', color=COLORS['text'])
                ax1.set_xlabel("Index", fontsize=8, color=COLORS['text'])
                ax1.set_ylabel("Prix (USD)", fontsize=8, color=COLORS['text'])
                ax1.legend(fontsize=7)
                ax1.tick_params(labelsize=7)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.grid(True, alpha=0.3, linestyle='--')

                # Train
                if y_true_train is not None and y_pred_train is not None:
                    ax2.plot(y_true_train, label="Réel", color=COLORS['success'], linewidth=1.5)
                    ax2.plot(y_pred_train, label="Prédit", color=COLORS['warning'], linewidth=1.5, alpha=0.8)
                    ax2.set_title("Train", fontsize=10, fontweight='bold', color=COLORS['text'])
                    ax2.set_xlabel("Index", fontsize=8, color=COLORS['text'])
                    ax2.set_ylabel("Prix (USD)", fontsize=8, color=COLORS['text'])
                    ax2.legend(fontsize=7)
                    ax2.tick_params(labelsize=7)
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    ax2.grid(True, alpha=0.3, linestyle='--')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


def display_model_results(pkl_path: str = "data/model_results.pkl", model_filter: str = None) -> None:
    viewer = ModelResultsViewer(default_pkl_path=pkl_path)
    viewer.render(model_filter=model_filter)
