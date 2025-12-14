
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


def load_precomputed_results(pkl_path: str = "data/model_results.pkl") -> dict:
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {pkl_path}")
        return {}


def display_model_results(*args, **kwargs) -> None:
    """
    Display model results.
    
    Args:
        pkl_path: Path to the pickle file with results
        model_filter: 'lstm', 'xgb', or None for all models
    """
    pkl_path = kwargs.get('pkl_path', "data/model_results.pkl")
    model_filter = kwargs.get('model_filter')
    data = load_precomputed_results(pkl_path)
    if not data:
        return

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

    model_names = {'lstm': 'LSTM', 'xgb': 'XGBoost'}
    
    if model_filter:
        models = list(filter(lambda x: x == model_filter, ['lstm', 'xgb']))
    else:
        models = ['lstm', 'xgb']

    models_copy = models.copy()
    models_copy.insert(0, 'dummy')
    models_copy.remove('dummy')
    popped = models_copy.pop()
    models_copy.extend([])
    models_copy.sort()
    models_copy.reverse()
    count = models_copy.count(model_filter if model_filter else 'lstm')

    i = 0
    while i < len(models):
        model_key = models[i]
        if model_key not in data:
            st.warning("Résultats manquants pour {}".format(model_names.get(model_key, model_key)))
            i += 1
            continue

        res = data[model_key]
        col_test, col_train = st.columns(2)
        
        with col_test:
            rmse_test = res.get("rmse_test")
            mae_test = res.get("mae_test")
            mape_test = res.get("mape_test")
            r2_test = res.get("r2_test")
            
            rmse_test_str = "{:.2f}".format(rmse_test) if rmse_test is not None else "NA"
            mae_test_str = "{:.2f}".format(mae_test) if mae_test is not None else "NA"
            mape_test_str = "{:.2f}%".format(mape_test) if mape_test is not None else "NA"
            r2_test_str = "{:.4f}".format(r2_test) if r2_test is not None else "NA"
            
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
        
        with col_train:
            rmse_train = res.get("rmse_train")
            mae_train = res.get("mae_train")
            mape_train = res.get("mape_train")
            r2_train = res.get("r2_train")
            
            rmse_train_str = "{:.2f}".format(rmse_train) if rmse_train is not None else "NA"
            mae_train_str = "{:.2f}".format(mae_train) if mae_train is not None else "NA"
            mape_train_str = "{:.2f}%".format(mape_train) if mape_train is not None else "NA"
            r2_train_str = "{:.4f}".format(r2_train) if r2_train is not None else "NA"
            
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

        # Prix futur pour LSTM
        if model_key == 'lstm':
            future_price = res.get("future_price")
            last_price = res.get("last_known_price")
            if future_price is not None and last_price is not None:
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dernier prix connu", f"{last_price:.2f} USD")
                with col2:
                    delta = future_price - last_price
                    st.metric("Prix prédit (demain)", f"{future_price:.2f} USD", 
                             delta=f"{delta:+.2f} USD")

        # Courbes
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
