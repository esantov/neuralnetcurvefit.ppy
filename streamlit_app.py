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

st.set_page_config(layout="centered")
st.title("Neural Network + Parametric Curve Fitting")

# Initialize report storage
if "export_tables" not in st.session_state:
    st.session_state["export_tables"] = []
if "export_plots" not in st.session_state:
    st.session_state["export_plots"] = []
if "report_elements" not in st.session_state:
    st.session_state["report_elements"] = {}

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
if uploaded_file:
    # Load data
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select sheet", xls.sheet_names)
    df = pd.read_excel(xls, sheet_name=sheet_name)
    st.dataframe(df.head())

    # Column selection
    sample_col = st.selectbox("Sample column", df.columns)
    x_col = st.selectbox("X column", df.select_dtypes("number").columns)
    y_col = st.selectbox("Y column", df.select_dtypes("number").columns)

    samples = df[sample_col].dropna().unique().tolist()
    selected_samples = st.multiselect("Select samples", samples, default=samples)
    treat_as_replicates = st.checkbox("Treat as replicates", value=False)
    threshold = st.number_input("Y Threshold", value=float(df[y_col].mean()), step=0.1)

    df = df[df[sample_col].isin(selected_samples)]

    # Transformation
    st.sidebar.subheader("Data Transformation")
    transform_option = st.sidebar.selectbox("Transform Y data", [
        "None", "Log Transform", "Z-Score Normalization",
        "Min-Max Scaling", "Baseline subtraction"
    ])

    x_data = df[x_col].values.reshape(-1, 1)
    y_data = df[y_col].values.reshape(-1, 1)
    if transform_option == "Log Transform":
        y_data = np.log1p(y_data)
    elif transform_option == "Z-Score Normalization":
        y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    elif transform_option == "Min-Max Scaling":
        y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))
    elif transform_option == "Baseline subtraction":
        y_data = y_data - y_data[0]

    # Scale
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x_data)
    y_scaled = scaler_y.fit_transform(y_data)
    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled, y_scaled, test_size=0.2, random_state=42
    )

    hidden_layers = st.slider("Hidden layer size", 5, 200, 50)
    model_choice = st.radio("Choose model", [
        "Neural Network", "Exponential", "Gompertz", "4PL", "5PL", "Linear"
    ])

    # Model functions
    def linear(x,a,b): return a*x + b
    def exponential(x,a,b): return a*np.exp(b*x)
    def gompertz(x,a,b,c): return a*np.exp(-b*np.exp(-c*x))
    def four_pl(x,A,B,C,D): return D + (A-D)/(1 + (x/C)**B)
    def five_pl(x,A,B,C,D,F): return D + (A-D)/((1 + (x/C)**B)**F)

    def calculate_tt_and_stderr(fit_func, popt, pcov, thr, bounds):
        res = root_scalar(lambda x: fit_func(x,*popt) - thr, bracket=bounds)
        if res.converged:
            tt = res.root
            grad = np.zeros_like(popt)
            for i in range(len(popt)):
                dp = np.zeros_like(popt); dp[i]=1e-5
                grad[i] = (fit_func(tt,*(popt+dp)) - fit_func(tt,*(popt-dp))) / (2e-5)
            var = grad.T @ pcov @ grad
            return round(tt,4), round(np.sqrt(var),4) if var>0 else (round(tt,4),"N/A")
        return "N/A","N/A"

    # Prepare plotting
    x_full = np.linspace(x_scaled.min(), x_scaled.max(), 500).reshape(-1,1)
    x_full_inv = scaler_x.inverse_transform(x_full)
    fig, ax = plt.subplots()
    tt_rows = []

    if model_choice=="Neural Network" and treat_as_replicates:
        nn = MLPRegressor((hidden_layers,), max_iter=10000, early_stopping=True, random_state=42)
        nn.fit(X_train, y_train.ravel())
        y_fit_full = scaler_y.inverse_transform(nn.predict(x_full).reshape(-1,1))
        ax.plot(x_full_inv, y_fit_full, '-', label="NN Fit")
        fit_label="NN Fit"
    else:
        funcs = {
            "Linear": linear, "Exponential": exponential,
            "Gompertz": gompertz, "4PL": four_pl, "5PL": five_pl
        }
        for samp in selected_samples:
            sub = df[df[sample_col]==samp]
            x_vals,y_vals = sub[x_col].values, sub[y_col].values
            ax.plot(x_vals, y_vals, 'o', label=samp)
            try:
                popt,pcov = curve_fit(funcs[model_choice], x_vals, y_vals, maxfev=10000)
                y_fit = funcs[model_choice](x_full_inv.ravel(), *popt)
                ax.plot(x_full_inv, y_fit, '-', label=f"{samp} fit")
                tt, se = calculate_tt_and_stderr(funcs[model_choice], popt, pcov, threshold, [x_vals.min(), x_vals.max()])
                tt_rows.append({"Sample":samp, "Tt":tt, "StdErr":se})
                fit_label=f"{model_choice} Fit"
            except:
                tt_rows.append({"Sample":samp, "Tt":"N/A", "StdErr":"N/A"})

    ax.axhline(threshold, linestyle='--', color='red', label='Threshold')
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(fit_label)
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    st.pyplot(fig)

    # Add-to-report buttons
    if st.button("➕ Add Plot to Report"):
        buf = io.BytesIO(); fig.savefig(buf,format="png",dpi=300,bbox_inches='tight')
        st.session_state["export_plots"].append((f"Plot_{len(st.session_state['export_plots'])+1}", buf.getvalue()))

    tt_df = pd.DataFrame(tt_rows)
    st.subheader("Threshold Time Results")
    st.dataframe(tt_df)
    if st.button("➕ Add Tt Table to Report"):
        st.session_state["export_tables"].append((f"Tt_Table_{len(st.session_state['export_tables'])+1}", tt_df.copy()))

    # Report selection checkboxes
    st.subheader("Select report contents")
    for name,_ in st.session_state["export_tables"]:
        st.session_state["report_elements"][name] = st.checkbox(name, value=True)
    for name,_ in st.session_state["export_plots"]:
        st.session_state["report_elements"][name] = st.checkbox(name, value=True)

    # Export final report
    if st.button("📥 Export Report as Excel"):
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            used = set()
            for name, tbl in st.session_state["export_tables"]:
                if st.session_state["report_elements"].get(name):
                    sheet = name[:31]
                    i=1
                    while sheet in used:
                        sheet = f"{name[:28]}_{i}"; i+=1
                    tbl.to_excel(writer, sheet_name=sheet, index=False)
                    used.add(sheet)
            for name, img in st.session_state["export_plots"]:
                if st.session_state["report_elements"].get(name):
                    sheet = name[:31]; i=1
                    while sheet in used:
                        sheet = f"{name[:28]}_{i}"; i+=1
                    ws = writer.book.add_worksheet(sheet)
                    ws.insert_image("B2", f"{sheet}.png", {"image_data":io.BytesIO(img)})
                    used.add(sheet)
        out.seek(0)
        st.download_button(
            "📥 Download Excel Report",
            data=out,
            file_name="curve_fit_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
