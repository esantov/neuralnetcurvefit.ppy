import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
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
        "Neural Network", "Exponential", "Gompertz", "4PL", "5PL", "Linear"
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
                        st.warning("⚠️ Logistic B model failed to converge.")
                        continue
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

    def linear(x, a, b):
    return a * x + b
    return a * x + b

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

    fit_label = None
    x_full = np.linspace(x_scaled.min(), x_scaled.max(), 1000).reshape(-1, 1)
    x_full_inv = scaler_x.inverse_transform(x_full)

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

        # Optionally calculate x for a given y
    st.markdown("### Predict X for given Y")
    input_y = st.number_input("Enter Y value to estimate X", value=float(df[y_col].mean()))
    x_for_y = "N/A"
    y_full_pred_func = None
    if fit_label and 'popt' in locals():
        if model_option == "Exponential":
            y_full_pred_func = lambda x: exponential(x, *popt)
        elif model_option == "Gompertz":
            y_full_pred_func = lambda x: gompertz(x, *popt)
        elif model_option == "4PL":
            y_full_pred_func = lambda x: four_pl(x, *popt)
        elif model_option == "5PL":
            y_full_pred_func = lambda x: five_pl(x, *popt)
        elif model_option == "Linear":
            y_full_pred_func = lambda x: popt[0] * x + popt[1]
        try:
            from scipy.optimize import root_scalar
            def root_func(x):
                return y_full_pred_func(x) - input_y
            bracket = [float(x_full_inv.min()), float(x_full_inv.max())]
            result = root_scalar(root_func, bracket=bracket)
            if result.converged:
                x_for_y = round(result.root, 4)
        except Exception as e:
            x_for_y = f"Error: {e}"
    st.write(f"**Estimated X for Y = {input_y}:** {x_for_y}")

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

    # Export plot as PNG

    # Export prediction table
    st.markdown("### Batch Y → X Table")
    y_values = st.text_area("Enter Y values (comma-separated)", value="0.2, 0.5, 0.8")
    x_estimates = []
    if fit_label and 'popt' in locals():
        try:
            y_vals = [float(val.strip()) for val in y_values.split(',') if val.strip() != '']
            for val in y_vals:
                def root_func(x): return model_func(x, *popt) - val
                from scipy.optimize import root_scalar
                try:
                    result = root_scalar(root_func, bracket=[float(x_full_inv.min()), float(x_full_inv.max())])
                    if result.converged:
                        x_estimates.append((val, round(result.root, 4)))
                    else:
                        x_estimates.append((val, 'N/A'))
                except:
                    x_estimates.append((val, 'Error'))
            x_table = pd.DataFrame(x_estimates, columns=["Y Value", "Estimated X"])
            st.dataframe(x_table)

            if st.button("Add X-for-Y Table to Export"):
                if 'export_tables' not in st.session_state:
                    st.session_state['export_tables'] = []
                st.session_state['export_tables'].append(("Y→X Table", x_table.copy()))
                st.success("Added to export queue!")
        except:
            st.warning("Invalid Y values")
    plot_buf = io.BytesIO()
    fig.savefig(plot_buf, format="png")
    if 'export_tables' not in st.session_state:
        st.session_state['export_tables'] = []
    plot_buf.seek(0)
    st.download_button(
        label="📥 Download Plot as PNG",
        data=plot_buf,
        file_name="fitted_plot.png",
        mime="image/png"
    )

    if st.button("📥 Export All Tables to Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for name, table in st.session_state['export_tables']:
                table.to_excel(writer, sheet_name=name[:31], index=False)
        output.seek(0)
        st.download_button(
            label="Download All Tables as Excel",
            data=output,
            file_name="fitting_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
