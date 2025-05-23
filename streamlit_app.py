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

st.set_page_config(
    layout="centered", 
    page_title="Neural Network Curve Fitting",
    page_icon="📈"
)


# ========== Initialize state ==========
if "export_tables" not in st.session_state:
    st.session_state["export_tables"] = []
if "export_plots" not in st.session_state:
    st.session_state["export_plots"] = []

# ========== Upload ==========
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select sheet", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    st.dataframe(df.head())

    sample_col = st.selectbox("Sample column", df.columns)
    x_col = st.selectbox("X column", df.select_dtypes("number").columns)
    y_col = st.selectbox("Y column", df.select_dtypes("number").columns)

    samples = df[sample_col].dropna().unique().tolist()
    selected_samples = st.multiselect("Select samples", samples, default=samples)
    treat_as_replicates = st.checkbox("Treat as replicates", value=False)
    threshold_value = st.number_input("Enter Y-axis threshold", value=float(df[y_col].mean()))

    df = df[df[sample_col].isin(selected_samples)]

    x_data = df[x_col].values.reshape(-1, 1)
    y_data = df[y_col].values.reshape(-1, 1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x_data)
    y_scaled = scaler_y.fit_transform(y_data)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

    # ===== Define models =====
    def linear(x, a, b): return a * x + b
    def exponential(x, a, b): return a * np.exp(b * x)
    def gompertz(x, a, b, c): return a * np.exp(-b * np.exp(-c * x))
    def four_pl(x, A, B, C, D): return D + (A - D) / (1 + (x / C) ** B)
    def five_pl(x, A, B, C, D, F): return D + (A - D) / ((1 + (x / C) ** B) ** F)

    def calculate_tt_and_stderr(fit_func, popt, pcov, threshold_y, x_bounds):
        def root_fn(x): return fit_func(x, *popt) - threshold_y
        result = root_scalar(root_fn, bracket=x_bounds)
        if result.converged:
            tt = result.root
            grad = np.zeros_like(popt)
            for i in range(len(popt)):
                dp = np.zeros_like(popt)
                dp[i] = 1e-5
                grad[i] = (fit_func(tt, *(popt + dp)) - fit_func(tt, *(popt - dp))) / (2e-5)
            tt_var = grad.T @ pcov @ grad
            tt_stderr = np.sqrt(tt_var) if tt_var > 0 else np.nan
            return round(tt, 4), round(tt_stderr, 4)
        else:
            return "N/A", "N/A"

    # ===== Model selection =====
    model_options = ["Neural Network", "Exponential", "Gompertz", "4PL", "5PL", "Linear"]
    model_choice = st.radio("Choose model", model_options)
    hidden_layers = st.slider("Hidden layers (for NN)", 5, 200, 50)

    # ===== Fitting =====
    x_full = np.linspace(x_scaled.min(), x_scaled.max(), 500).reshape(-1, 1)
    x_full_inv = scaler_x.inverse_transform(x_full)
    fig, ax = plt.subplots()
    tt_rows = []

    if model_choice == "Neural Network" and treat_as_replicates:
        nn = MLPRegressor((hidden_layers,), max_iter=10000, early_stopping=True)
        nn.fit(X_train, y_train.ravel())
        y_pred_full = nn.predict(x_full)
        y_full = scaler_y.inverse_transform(y_pred_full.reshape(-1, 1))
        ax.plot(x_full_inv, y_full, '-', label="Neural Network Fit")
        fit_func = lambda x: scaler_y.inverse_transform(nn.predict(scaler_x.transform(np.array(x).reshape(-1, 1))).reshape(-1, 1)).ravel()[0]
        try:
            def root_fn(x): return fit_func([x]) - threshold_value
            result = root_scalar(root_fn, bracket=[x_data.min(), x_data.max()])
            if result.converged:
                tt = round(result.root, 4)
                tt_rows.append({"Sample": "Replicates", "Tt": tt, "StdErr": "N/A"})
        except:
            tt_rows.append({"Sample": "Replicates", "Tt": "N/A", "StdErr": "N/A"})

    else:
        model_funcs = {
            "Linear": linear, "Exponential": exponential, "Gompertz": gompertz,
            "4PL": four_pl, "5PL": five_pl
        }
        for sample in selected_samples:
            sub = df[df[sample_col] == sample]
            x_vals = sub[x_col].values
            y_vals = sub[y_col].values
            ax.plot(x_vals, y_vals, 'o', label=f"{sample} data")

            try:
                popt, pcov = curve_fit(model_funcs[model_choice], x_vals, y_vals, maxfev=10000)
                y_fit = model_funcs[model_choice](x_full_inv.ravel(), *popt)
                ax.plot(x_full_inv, y_fit, '-', label=f"{sample} fit")
                tt, stderr = calculate_tt_and_stderr(model_funcs[model_choice], popt, pcov, threshold_value, [x_vals.min(), x_vals.max()])
                tt_rows.append({"Sample": sample, "Tt": tt, "StdErr": stderr})
            except:
                tt_rows.append({"Sample": sample, "Tt": "N/A", "StdErr": "N/A"})

    ax.axhline(threshold_value, color='red', linestyle='--', label='Threshold')
    ax.set_title(f"Fitting: {model_choice}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize='small'
    )

    st.pyplot(fig)

    # ===== Add to report buttons =====
    if st.button("📊 Add Plot to Report"):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.session_state["export_plots"].append(("Fitting Plot", buf.getvalue()))
        st.success("Plot added to report")

    tt_df = pd.DataFrame(tt_rows)
    st.write("### Tt (Threshold Time) Summary")
    st.dataframe(tt_df)

    if st.button("📑 Add Tt Table to Report"):
        st.session_state["export_tables"].append(("Tt Summary", tt_df.copy()))
        st.success("Tt data added to report")

    st.write("---")
    export_selections = st.multiselect("Select report contents to export", [n for n, _ in st.session_state["export_tables"]] + [n for n, _ in st.session_state["export_plots"]])

    if st.button("📥 Export Report as Excel"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            used_names = set()
            def unique_name(base):
                i = 1
                name = base[:31]
                while name in used_names:
                    name = f"{base[:28]}_{i}"
                    i += 1
                used_names.add(name)
                return name
    
            for name, table in st.session_state["export_tables"]:
                if name in export_selections:
                    safe_name = unique_name(name)
                    table.to_excel(writer, sheet_name=safe_name, index=False)
    
            for name, img_bytes in st.session_state["export_plots"]:
                if name in export_selections:
                    safe_name = unique_name(name)
                    worksheet = writer.book.add_worksheet(safe_name)
                    img_stream = io.BytesIO(img_bytes)
                    worksheet.insert_image("B2", f"{safe_name}.png", {"image_data": img_stream})
    if st.button("📥 Export Full Report (Full Data + Fit Summary)"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Sheet: Original Data
            df.to_excel(writer, sheet_name="Original Data", index=False)
    
            # Sheet: Fitting Parameters Summary
            if 'fit_summary_table' in st.session_state:
                st.session_state['fit_summary_table'].to_excel(writer, sheet_name="Fitting Summary", index=False)
    
            # Sheet: Y→X Predictions
            if 'yx_predictions' in st.session_state:
                st.session_state['yx_predictions'].to_excel(writer, sheet_name="Y→X Lookup", index=False)
    
            # Sheet: Fitted Curve Values
            if 'fit_curve_data' in st.session_state:
                st.session_state['fit_curve_data'].to_excel(writer, sheet_name="Fitted Curve", index=False)
    
            # Add plots (if desired)
            workbook = writer.book
            for idx, (plot_title, img_bytes) in enumerate(st.session_state.get("export_plots", [])):
                worksheet_name = f"Plot_{idx+1}"[:31]
                worksheet = workbook.add_worksheet(worksheet_name)
                worksheet.insert_image('B2', f"{plot_title}.png", {'image_data': io.BytesIO(img_bytes)})
    
        output.seek(0)
        st.download_button(
            label="Download Excel Report",
            data=output,
            file_name="curve_fit_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
            output.seek(0)
            st.download_button("📥 Download Full Report", output, file_name="Final_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.download_button(
    label="📥 Download Full Curve Fit Report",
    data=output_bytes,
    file_name="curve_fit_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
