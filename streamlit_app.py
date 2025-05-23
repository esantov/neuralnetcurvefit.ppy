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
    treat_as_replicates = st.checkbox("Treat all selected samples as replicates (fit together)", value=False)
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
    model_options = [
        "Neural Network", "Exponential", "Gompertz", "4PL",
        "Sigmoid B", "Sigmoid B Modified", "Logistic A", "Logistic B",
        "Don Levin", "5PL", "4PL 2D", "Generalised Logistic",
        "Gompertz A", "Gompertz B", "Gompertz C", "Hill",
        "Jacquelin", "Janoschek", "Janoschek Modified",
        "Richards", "Sigmoid A", "Sigmoid A Modified"
    ]

    auto_suggest = st.checkbox("Suggest best model automatically")
    if auto_suggest:
        best_r2 = -np.inf
        best_model = None
        for model in model_options:
            try:
                x = scaler_x.inverse_transform(X_train)
                y = scaler_y.inverse_transform(y_train)
                if model == "Exponential":
                    try:
                        popt, _ = curve_fit(exponential, x.ravel(), y.ravel(), maxfev=10000)
                        y_pred = exponential(scaler_x.inverse_transform(X_test).ravel(), *popt)
                    except RuntimeError:
                        st.warning("⚠️ Model failed to converge.")
                        continue
                elif model == "Gompertz":
                    try:
                        popt, _ = curve_fit(gompertz, x.ravel(), y.ravel(), maxfev=10000)
                        y_pred = gompertz(scaler_x.inverse_transform(X_test).ravel(), *popt)
                    except RuntimeError:
                        st.warning("⚠️ Gompertz model failed to converge.")
                        continue
                elif model == "4PL":
                    try:
                        popt, _ = curve_fit(four_pl, x.ravel(), y.ravel(), maxfev=10000)
                        y_pred = four_pl(scaler_x.inverse_transform(X_test).ravel(), *popt)
                    except RuntimeError:
                        st.warning("⚠️ 4PL model failed to converge.")
                        continue
                elif model == "Sigmoid B":
                    try:
                        popt, _ = curve_fit(sigmoid_b, x.ravel(), y.ravel(), maxfev=10000)
                        y_pred = sigmoid_b(scaler_x.inverse_transform(X_test).ravel(), *popt)
                    except RuntimeError:
                        st.warning("⚠️ Sigmoid B model failed to converge.")
                        continue
                elif model == "Logistic B":
                    try:
                        popt, _ = curve_fit(logistic_b, x.ravel(), y.ravel(), maxfev=10000)
                        y_pred = logistic_b(scaler_x.inverse_transform(X_test).ravel(), *popt)
                    except RuntimeError:
                        st.warning("⚠️ Logistic B model failed to converge.")
                        continue(scaler_x.inverse_transform(X_test).ravel(), *popt)
                else:
                    continue
                y_test_inv = scaler_y.inverse_transform(y_test)
                y_pred_inv = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1))
                r2 = r2_score(y_test_inv, y_pred_inv)
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
            except:
                continue
        model_option = best_model if best_model else "Exponential"
        st.success(f"Best model by R²: {model_option} ({best_r2:.4f})")
    else:
        model_option = st.radio("Choose model to fit", model_options)

    def exponential(x, a, b):
        return a * np.exp(b * x)

    def gompertz(x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))

    def four_pl(x, A, B, C, D):
        return D + (A - D) / (1 + (x / C) ** B)

    def five_pl(x, A, B, C, D, F):
        return D + (A - D) / ((1 + (x / C) ** B) ** F)

    def sigmoid_b(x, a, b, c):
        return a / (1.0 + np.exp(-(x - b) / c))

    def sigmoid_b_mod(x, a, b, c, d):
        return a / (1.0 + np.exp(-(x - b) / c)) ** d

    def logistic_a(x, a, b, c):
        return a / (1.0 + b * np.exp(-c * x))

    def logistic_b(x, a, b, c):
        return a / (1.0 + (x / b) ** c)

    def sigmoid_a(x, a, b):
        return 1.0 / (1.0 + np.exp(-a * (x - b)))

    def sigmoid_a_mod(x, a, b, c):
        return 1.0 / (1.0 + np.exp(-a * (x - b))) ** c

    def richards(x, a, b, c, d):
        return 1.0 / (a + b * np.exp(c * x)) ** d

    def janoschek(x, a, b, c):
        return a - (1.0 - np.exp(-b * x ** c))

    def janoschek_mod(x, a, w0, b, c):
        return a - (a - w0) * (1.0 - np.exp(-b * x ** c))

    def hill(x, a, b, c):
        return a * x ** b / (c ** b + x ** b)

    def gompertz_a(x, a, b, c):
        return a * np.exp(-np.exp(b - c * x))

    def gompertz_b(x, a, b, c):
        return a * np.exp(-np.exp((x - b) / c))

    def gompertz_c(x, a, b, c):
        return a * np.exp(b * np.exp(c * x))

    def generalised_logistic(x, A, C, T, B, M):
        return A + C / ((1 + T * np.exp(-B * (x - M))) ** (1 / T))

    def jacquelin(x, L, b, k, c, h):
        return L / (1.0 + b * np.exp(-k * x) + c * np.exp(h * x))

    def don_levin(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
        return (
            a1 / (1.0 + np.exp(-(x - b1) / c1)) +
            a2 / (1.0 + np.exp(-(x - b2) / c2)) +
            a3 / (1.0 + np.exp(-(x - b3) / c3))
        )

    def sigmoid_b(x, a, b, c):
        return a / (1.0 + np.exp(-(x - b) / c))

    def sigmoid_b_mod(x, a, b, c, d):
        return a / (1.0 + np.exp(-(x - b) / c)) ** d

    def logistic_a(x, a, b, c):
        return a / (1.0 + b * np.exp(-c * x))

    def logistic_b(x, a, b, c):
        return a / (1.0 + (x / b) ** c)

    fit_label = None
    x_full = np.linspace(x_scaled.min(), x_scaled.max(), 1000).reshape(-1, 1)

    if model_option == "Neural Network":
        if treat_as_replicates:
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
    elif treat_as_replicates:
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
        elif model_option == "Sigmoid B":
            popt, _ = curve_fit(sigmoid_b, x.ravel(), y.ravel(), maxfev=10000)
            y_pred = sigmoid_b(scaler_x.inverse_transform(X_test).ravel(), *popt)
            y_full_pred = sigmoid_b(scaler_x.inverse_transform(x_full).ravel(), *popt)
            fit_label = "Sigmoid B Fit"
        elif model_option == "Sigmoid B Modified":
            popt, _ = curve_fit(sigmoid_b_mod, x.ravel(), y.ravel(), maxfev=10000)
            y_pred = sigmoid_b_mod(scaler_x.inverse_transform(X_test).ravel(), *popt)
            y_full_pred = sigmoid_b_mod(scaler_x.inverse_transform(x_full).ravel(), *popt)
            fit_label = "Sigmoid B Mod Fit"
        elif model_option == "Logistic A":
            popt, _ = curve_fit(logistic_a, x.ravel(), y.ravel(), maxfev=10000)
            y_pred = logistic_a(scaler_x.inverse_transform(X_test).ravel(), *popt)
            y_full_pred = logistic_a(scaler_x.inverse_transform(x_full).ravel(), *popt)
            fit_label = "Logistic A Fit"
        elif model_option == "Logistic B":
            popt, _ = curve_fit(logistic_b, x.ravel(), y.ravel(), maxfev=10000)
            y_pred = logistic_b(scaler_x.inverse_transform(X_test).ravel(), *popt)
            y_full_pred = logistic_b(scaler_x.inverse_transform(x_full).ravel(), *popt)
            fit_label = "Logistic B Fit"
            popt, _ = curve_fit(four_pl, x.ravel(), y.ravel(), maxfev=10000)
            y_pred = four_pl(scaler_x.inverse_transform(X_test).ravel(), *popt)
            y_full_pred = four_pl(scaler_x.inverse_transform(x_full).ravel(), *popt)
            fit_label = "4PL Fit"

    # Inverse transform and plot
    x_full_inv = scaler_x.inverse_transform(x_full)
    if model_option == "Neural Network":
        y_full_inv = scaler_y.inverse_transform(np.array(y_full_pred).reshape(-1, 1))
    else:
        y_full_inv = np.array(y_full_pred).reshape(-1, 1)

    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_col], 'o', label='Original')
    ax.plot(x_full_inv, y_full_inv, '-', label=fit_label)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{fit_label}")
    ax.legend()
    st.pyplot(fig)
