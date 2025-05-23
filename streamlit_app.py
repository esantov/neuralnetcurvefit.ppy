import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

    # Neural network regressor
    hidden_layers = st.slider("Hidden layer sizes", 1, 100, 50)
    nn = MLPRegressor(
        hidden_layer_sizes=(hidden_layers, hidden_layers),  # deeper net
        activation='tanh',
        solver='adam',
        learning_rate='adaptive',
        early_stopping=True,
        max_iter=10000,
        random_state=42
    )
    nn.fit(X_train, y_train.ravel())

    # Predict
    y_pred = nn.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test)

    r2 = r2_score(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    st.write(f"**R²:** {r2:.4f}, **RMSE:** {rmse:.4f}")

    # Plot
    fig, ax = plt.subplots()
    x_full = np.linspace(x_scaled.min(), x_scaled.max(), 1000).reshape(-1, 1)
    y_full_pred = nn.predict(x_full)
    x_full_inv = scaler_x.inverse_transform(x_full)
    y_full_inv = scaler_y.inverse_transform(y_full_pred.reshape(-1, 1))

    ax.plot(df[x_col], df[y_col], 'o', label='Original')
    ax.plot(x_full_inv, y_full_inv, '-', label='NN Fit')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    st.pyplot(fig)

    # Optional: Save model output
    if st.button("Download Fit Data"):
        fit_df = pd.DataFrame({x_col: x_full_inv.flatten(), f"Fitted_{y_col}": y_full_inv.flatten()})
        st.download_button("Download CSV", fit_df.to_csv(index=False), file_name="nn_fitted_curve.csv")
