
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle


def load_precomputed_results(pkl_path: str = "data/model_results.pkl") -> dict:
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {pkl_path}")
        return {}


def display_model_results(pkl_path: str = "data/model_results.pkl") -> None:
    
    st.subheader("Résultats des Modèles de Prédiction")

    data = load_precomputed_results(pkl_path)
    if not data:
        return

    models = ['lstm', 'xgb']
    model_names = {'lstm': 'LSTM', 'xgb': 'XGBoost'}

    for model_key in models:
        if model_key not in data:
            st.warning(f"Résultats manquants pour {model_names[model_key]}")
            continue

        res = data[model_key]
        st.subheader(f"Résultats {model_names[model_key]}")

        # Métriques
        rmse_test = res.get("rmse_test")
        mae_test = res.get("mae_test")
        mape_test = res.get("mape_test")
        r2_test = res.get("r2_test")
        rmse_train = res.get("rmse_train")
        mae_train = res.get("mae_train")
        mape_train = res.get("mape_train")
        r2_train = res.get("r2_train")

        st.write("**Métriques Test :**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{rmse_test:.2f}" if rmse_test is not None else "NA")
        col2.metric("MAE", f"{mae_test:.2f}" if mae_test is not None else "NA")
        col3.metric("MAPE", f"{mape_test:.2f}%" if mape_test is not None else "NA")
        col4.metric("R²", f"{r2_test:.4f}" if r2_test is not None else "NA")

        st.write("**Métriques Train :**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{rmse_train:.2f}" if rmse_train is not None else "NA")
        col2.metric("MAE", f"{mae_train:.2f}" if mae_train is not None else "NA")
        col3.metric("MAPE", f"{mape_train:.2f}%" if mape_train is not None else "NA")
        col4.metric("R²", f"{r2_train:.4f}" if r2_train is not None else "NA")

        # Prix futur pour LSTM
        if model_key == 'lstm':
            future_price = res.get("future_price")
            last_price = res.get("last_known_price")
            if future_price is not None:
                st.success(f"Dernier prix connu : {last_price:.2f} USD")
                st.success(f"Prix prédit pour demain : {future_price:.2f} USD")

        # Courbes
        y_true_test = res.get("y_test_true")
        y_pred_test = res.get("y_test_pred")
        y_true_train = res.get("y_train_true")
        y_pred_train = res.get("y_train_pred")

        if y_true_test is not None and y_pred_test is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Test
            ax1.plot(y_true_test, label="Réel (test)", color="blue")
            ax1.plot(y_pred_test, label=f"{model_names[model_key]} (test)", color="red", alpha=0.7)
            ax1.set_title(f"{model_names[model_key]} vs Réel - Test")
            ax1.set_xlabel("Index temporel")
            ax1.set_ylabel("Prix de Clôture (USD)")
            ax1.legend()
            
            # Train (si disponible)
            if y_true_train is not None and y_pred_train is not None:
                ax2.plot(y_true_train, label="Réel (train)", color="green")
                ax2.plot(y_pred_train, label=f"{model_names[model_key]} (train)", color="orange", alpha=0.7)
                ax2.set_title(f"{model_names[model_key]} vs Réel - Train")
                ax2.set_xlabel("Index temporel")
                ax2.set_ylabel("Prix de Clôture (USD)")
                ax2.legend()
            
            st.pyplot(fig)
