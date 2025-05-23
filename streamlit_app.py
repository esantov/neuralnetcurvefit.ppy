import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit, root_scalar

st.title("Neural Network Curve Fitting")

if "export_tables" not in st.session_state:
    st.session_state["export_tables"] = []
if "export_plots" not in st.session_state:
    st.session_state["export_plots"] = []
if "report_elements" not in st.session_state:
    st.session_state["report_elements"] = {}

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

def linear(x, a, b): return a * x + b
def exponential(x, a, b): return a * np.exp(b * x)
def gompertz(x, a, b, c): return a * np.exp(-b * np.exp(-c * x))
def four_pl(x, A, B, C, D): return D + (A - D) / (1 + (x / C) ** B)
def five_pl(x, A, B, C, D, F): return D + (A - D) / ((1 + (x / C) ** B) ** F)

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

    threshold = st.number_input("Enter Y threshold", value=float(df[y_col].mean()), step=0.1)

    df = df[df[sample_col].isin(selected_samples)]
    colors = plt.cm.get_cmap('tab10', len(selected_samples))

    hidden_layers = st.slider("Hidden layer size", min_value=5, max_value=200, value=50)
    model_option = st.selectbox("Choose model", ["Neural Network", "Exponential", "Gompertz", "4PL", "5PL", "Linear"])

    x_data = df[x_col].values.reshape(-1, 1)
    y_data = df[y_col].values.reshape(-1, 1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x_data)
    y_scaled = scaler_y.fit_transform(y_data)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

    fit_label = None
    y_full_pred_func = None
    x_full = np.linspace(x_scaled.min(), x_scaled.max(), 1000).reshape(-1, 1)
    x_full_inv = scaler_x.inverse_transform(x_full)

    fig, ax = plt.subplots()
    for i, sample in enumerate(selected_samples):
        sub_df = df[df[sample_col] == sample]
        ax.plot(sub_df[x_col], sub_df[y_col], 'o', label=f"{sample}", color=colors(i))

    if model_option == "Neural Network" and treat_as_replicates:
        nn = MLPRegressor(hidden_layer_sizes=(hidden_layers,), max_iter=10000, early_stopping=True, random_state=42)
        nn.fit(X_train, y_train.ravel())
        y_full_pred = nn.predict(x_full)
        y_full_pred_func = lambda x: scaler_y.inverse_transform(nn.predict(scaler_x.transform(np.array(x).reshape(-1, 1))).reshape(-1, 1)).ravel()
        ax.plot(x_full_inv, scaler_y.inverse_transform(y_full_pred.reshape(-1, 1)), '-', color='black', label="NN Fit")
        fit_label = "NN Fit"
    elif treat_as_replicates:
        x = scaler_x.inverse_transform(X_train)
        y = scaler_y.inverse_transform(y_train)
        try:
            if model_option == "Exponential":
                popt, pcov = curve_fit(exponential, x.ravel(), y.ravel(), maxfev=10000)
                model_func = exponential
            elif model_option == "Gompertz":
                popt, pcov = curve_fit(gompertz, x.ravel(), y.ravel(), maxfev=10000)
                model_func = gompertz
            elif model_option == "4PL":
                popt, pcov = curve_fit(four_pl, x.ravel(), y.ravel(), maxfev=10000)
                model_func = four_pl
            elif model_option == "5PL":
                popt, pcov = curve_fit(five_pl, x.ravel(), y.ravel(), maxfev=10000)
                model_func = five_pl
            elif model_option == "Linear":
                popt, pcov = curve_fit(linear, x.ravel(), y.ravel(), maxfev=10000)
                model_func = linear

            y_full_pred = model_func(x_full_inv.ravel(), *popt)
            ax.plot(x_full_inv, y_full_pred, '-', color='black', label=f"{model_option} Fit")
            y_full_pred_func = lambda x: model_func(np.array(x), *popt)
            fit_label = f"{model_option} Fit"

            # Estimate Tt
            def root_func(x): return y_full_pred_func([x])[0] - threshold
            result = root_scalar(root_func, bracket=[float(x_full_inv.min()), float(x_full_inv.max())])
            stderr = np.sqrt(np.sum(np.diag(pcov))) if 'pcov' in locals() and pcov.size > 0 else "N/A"
            if result.converged:
                tt_result = pd.DataFrame({
                    "Threshold": [threshold],
                    "Tt (Threshold Time)": [round(result.root, 4)],
                    "Std Error": [round(stderr, 4) if isinstance(stderr, float) else stderr]
                })
                st.dataframe(tt_result)

                if st.button("Add Data to Report"):
                    st.session_state["export_tables"].append(("Tt Results", tt_result.copy()))
                    st.session_state["report_elements"]["Tt Results"] = True

        except Exception as e:
            st.warning(f"Model fitting failed: {e}")

    ax.axhline(threshold, linestyle='--', color='gray', label="Threshold")
    ax.set_title(f"{fit_label}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    st.pyplot(fig)

    if st.button("Add Plot to Report"):
        plot_buf = io.BytesIO()
        fig.savefig(plot_buf, format="png", dpi=300)
        plot_buf.seek(0)
        st.session_state["export_plots"].append(("Fitting Plot", plot_buf.read()))
        st.session_state["report_elements"]["Fitting Plot"] = True

    st.subheader("Select items to include in final report")
    for key in st.session_state["report_elements"]:
        st.session_state["report_elements"][key] = st.checkbox(f"Include: {key}", value=st.session_state["report_elements"][key])

    if st.button("📥 Export Report as Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for name, table in st.session_state.get("export_tables", []):
                if st.session_state["report_elements"].get(name):
                    table.to_excel(writer, sheet_name=name[:31], index=False)
            for name, img_bytes in st.session_state.get("export_plots", []):
                if st.session_state["report_elements"].get(name):
                    worksheet = writer.book.add_worksheet(name[:31])
                    img_stream = io.BytesIO(img_bytes)
                    worksheet.insert_image("B2", f"{name}.png", {"image_data": img_stream})
        output.seek(0)
        st.download_button("📤 Download Excel Report", data=output, file_name="fitting_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
import xlsxwriter

if 'report_data' not in st.session_state:
    st.session_state['report_data'] = []

# Button to save current fit and original data
if st.button("Add Current Data to Report"):
    report_entry = {
        "label": fit_label,
        "original": df[[x_col, y_col]],
        "summary": pd.DataFrame([{
            "Model": model_option,
            "R²": round(r2_score(scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1))), 4) if model_option != "Neural Network" else "N/A",
            "RMSE": round(np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test), scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)))), 4) if model_option != "Neural Network" else "N/A",
            "Parameters": str(popt) if 'popt' in locals() else "N/A",
            "Equation": fit_label,
            "Inverse": "Defined in logic"  # Replace with actual inverse if available
        }])
    }
    st.session_state['report_data'].append(report_entry)
    st.success("Added to report queue")

# Export full report to Excel
if st.button("📥 Export Final Report (XLSX)"):
    final_buf = io.BytesIO()
    with pd.ExcelWriter(final_buf, engine='xlsxwriter') as writer:
        for idx, entry in enumerate(st.session_state['report_data']):
            sheet_name = f"Sample {idx+1} - {entry['label'][:20]}"
            entry['original'].to_excel(writer, sheet_name + " Raw", index=False)
            entry['summary'].to_excel(writer, sheet_name + " Summary", index=False)
    final_buf.seek(0)
    st.download_button(
        label="Download Full Report",
        data=final_buf,
        file_name="fitting_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
