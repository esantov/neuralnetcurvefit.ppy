import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

st.title("Neural Network Curve Fitting")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select sheet to analyze", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    st.write("### Data Preview", df.head())

    sample_col = st.selectbox("Select sample identifier column", df.columns)
    x_col = st.selectbox("Select X-axis column", df.select_dtypes(include='number').columns)
    y_col = st.selectbox("Select Y-axis column", df.select_dtypes(include='number').columns)
    samples = df[sample_col].dropna().unique().tolist()
    selected_samples = st.multiselect("Select samples to fit", samples, default=samples)
    df = df[df[sample_col].isin(selected_samples)]

    # Select columns
    x_col = st.selectbox("Select X column", df.columns)
    y_col = st.selectbox("Select Y column", df.columns)

    # Prepare data
    x_data = df[x_col].values.reshape(-1, 1)
    y_data = df[y_col].values.reshape(-1, 1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x_data)
    y_scaled = scaler_y.fit_transform(y_data)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

    # Select hidden layer size for NN
    hidden_layers = st.slider("Hidden layer size", min_value=5, max_value=200, value=50)

    # Neural network regressor
    model_option = st.radio("Choose model to fit", ["Neural Network", "Exponential", "Gompertz", "4PL"])

    def exponential(x, a, b):
        return a * np.exp(b * x)

    def gompertz(x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))

    def four_pl(x, A, B, C, D):
        return D + (A - D) / (1 + (x / C) ** B)

    fit_label = None
    x_full = np.linspace(x_scaled.min(), x_scaled.max(), 1000).reshape(-1, 1)

    if model_option == "Neural Network":
        nn = MLPRegressor(
            hidden_layer_sizes=(hidden_layers, hidden_layers),
            activation='tanh',
            solver='adam',
            learning_rate='adaptive',
            early_stopping=True,
            max_iter=10000,
            random_state=42
        )
        nn.fit(X_train, y_train.ravel())
        y_pred = nn.predict(X_test)
        y_full_pred = nn.predict(x_full)
        fit_label = "NN Fit"
    else:
        x = scaler_x.inverse_transform(X_train)
        y = scaler_y.inverse_transform(y_train)
        if model_option == "Exponential":
            popt, _ = curve_fit(exponential, x.ravel(), y.ravel(), maxfev=10000)
            y_pred = exponential(scaler_x.inverse_transform(X_test).ravel(), *popt)
            y_full_pred = exponential(scaler_x.inverse_transform(x_full).ravel(), *popt)
            fit_label = "Exponential Fit"
        elif model_option == "Gompertz":
            popt, _ = curve_fit(gompertz, x.ravel(), y.ravel(), maxfev=10000)
            y_pred = gompertz(scaler_x.inverse_transform(X_test).ravel(), *popt)
            y_full_pred = gompertz(scaler_x.inverse_transform(x_full).ravel(), *popt)
            fit_label = "Gompertz Fit"
        elif model_option == "4PL":
            popt, _ = curve_fit(four_pl, x.ravel(), y.ravel(), maxfev=10000)
            y_pred = four_pl(scaler_x.inverse_transform(X_test).ravel(), *popt)
            y_full_pred = four_pl(scaler_x.inverse_transform(x_full).ravel(), *popt)
            fit_label = "4PL Fit"
    # Predict
    if model_option == "Neural Network":
        y_pred = nn.predict(X_test)
        y_full_pred = nn.predict(x_full)
    else:
        y_pred = y_pred
        y_full_pred = y_full_pred
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test)

    r2 = r2_score(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    st.write(f"**R²:** {r2:.4f}, **RMSE:** {rmse:.4f}")

    # Domain-specific suggestion
    st.info("ℹ️ For microbial growth or respirometry data, exponential or sigmoid-like models (e.g., Gompertz, 4PL) are commonly used.")

    # Suggestion
    if r2 < 0.90:
        st.warning("⚠️ The neural network fit may not be optimal. Consider trying a parametric model (e.g., 4PL, Gompertz) or adjusting network architecture.")
    elif r2 < 0.95:
        st.info("ℹ️ Fit is acceptable but could be improved with tuning or alternative models.")
    else:
        st.success("✅ Excellent fit!")

    # Plot per sample
    for sample in selected_samples:
        fig, ax = plt.subplots()
        group = df[df[sample_col] == sample]
        x_sample = group[x_col].values.reshape(-1, 1)
        y_sample = group[y_col].values.reshape(-1, 1)

        x_scaled = scaler_x.fit_transform(x_sample)
        y_scaled = scaler_y.fit_transform(y_sample)

        x_full = np.linspace(x_scaled.min(), x_scaled.max(), 1000).reshape(-1, 1)

        if model_option == "Neural Network":
            nn = MLPRegressor(hidden_layer_sizes=(hidden_layers, hidden_layers), activation='tanh',
                             solver='adam', learning_rate='adaptive', early_stopping=True,
                             max_iter=10000, random_state=42)
            nn.fit(x_scaled, y_scaled.ravel())
            y_pred = nn.predict(x_full)
        else:
            x_inv = scaler_x.inverse_transform(x_scaled)
            y_inv = scaler_y.inverse_transform(y_scaled)
            if model_option == "Exponential":
                popt, _ = curve_fit(exponential, x_inv.ravel(), y_inv.ravel(), maxfev=10000)
                y_pred = exponential(scaler_x.inverse_transform(x_full).ravel(), *popt)
            elif model_option == "Gompertz":
                popt, _ = curve_fit(gompertz, x_inv.ravel(), y_inv.ravel(), maxfev=10000)
                y_pred = gompertz(scaler_x.inverse_transform(x_full).ravel(), *popt)
            elif model_option == "4PL":
                popt, _ = curve_fit(four_pl, x_inv.ravel(), y_inv.ravel(), maxfev=10000)
                y_pred = four_pl(scaler_x.inverse_transform(x_full).ravel(), *popt)

        x_plot = scaler_x.inverse_transform(x_full).ravel()
        if model_option == "Neural Network":
            y_plot = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        else:
            y_plot = y_pred

        ax.plot(group[x_col], group[y_col], 'o', label=f'{sample} Original')
        ax.plot(x_plot, y_plot, '-', label=f'{sample} Fit')
        ax.set_title(f"{sample} - {model_option} Fit")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        st.pyplot(fig)

    # Optional: Save model output
    if model_option != "Neural Network" and 'ci' in locals():
        fit_df = pd.DataFrame({
            x_col: x_full_inv.flatten(),
            f"Fitted_{y_col}": y_full_inv.flatten(),
            "Lower_CI": (y_full_inv.flatten() - ci),
            "Upper_CI": (y_full_inv.flatten() + ci)
        })
    else:
        fit_df = pd.DataFrame({x_col: x_full_inv.flatten(), f"Fitted_{y_col}": y_full_inv.flatten()})
    if st.button("Download Fit Data"):
        fit_df = pd.DataFrame({x_col: x_full_inv.flatten(), f"Fitted_{y_col}": y_full_inv.flatten()})
        st.download_button("Download CSV", fit_df.to_csv(index=False), file_name="nn_fitted_curve.csv")
