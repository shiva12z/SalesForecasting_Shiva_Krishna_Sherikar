# 📊 End-to-End Sales Forecasting & Demand Intelligence System

An intelligent sales forecasting, anomaly detection, and product demand segmentation pipeline built on 4 years (2015–2018) of historical Superstore retail transaction data and global video game industry sales benchmarks.

🔗 **Live Deployed Dashboard**: [Streamlit Cloud](https://shiva12z-salesforecasting-shiva-krishna-sherikar-app-hwrhvw.streamlit.app/)

---

## 🏗️ Codebase Structure

- **`analysis.ipynb`**: R&D Jupyter notebook covering EDA, stationarity testing, model training/backtesting (SARIMA, Prophet, XGBoost), anomaly detection comparison (Isolation Forest vs. Z-Score), and KMeans clustering.
- **`app.py`**: Streamlit interactive dashboard showcasing sales trends, interactive forecasts (Prophet with backtesting metrics), anomaly details, and product demand cluster profiles, rendered in a high-contrast premium light theme.
- **`summary.pdf`**: Generated business-ready executive PDF report compiling the end-to-end data cleaning details, forecasting performance comparison, anomaly insights, and inventory strategies.
- **`refactor_notebook.py`**: Utility script to automatically update the notebook to fix XGBoost data leakage, resolve CPU count warnings, and prevent cluster label clashes.
- **`requirements.txt`**: Package dependencies required to run the pipeline.
- **`train.csv`**: Core dataset containing transactional retail sales.
- **`vgsales.csv`**: Supplementary dataset containing historical global video game sales.

---

## 🛠️ Setup & Installation Instructions

Follow these steps to set up and run the project locally:

### 1. Prerequisite
Ensure you have **Python 3.10+** (Python 3.11 or 3.12 recommended) installed on your system.

### 2. Set Up a Virtual Environment (Optional but Recommended)
Open your terminal in the project directory and run:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (CMD/PowerShell):
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all package requirements listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Run Notebook Refactoring (Important)
Apply critical logical fixes (data leakage correction, Windows joblib warnings suppression, and cluster labeling overlap fixes) to `analysis.ipynb`:

```bash
python refactor_notebook.py
```

### 5. Run the Analysis Notebook
Open and execute the Jupyter notebook to view research results and train models:

```bash
jupyter notebook analysis.ipynb
```
*(Alternatively, you can open `analysis.ipynb` directly in VS Code or JupyterLab and select **Run All**).*

### 6. Launch the Streamlit Dashboard
Start the live interactive forecasting dashboard locally:

```bash
streamlit run app.py
```
*The local dashboard will automatically open in your default browser at `http://localhost:8501`. It is pre-configured with a clean, high-contrast light theme (`config.toml`) for perfect legibility across all charts and controls.*

---

## 🧼 Data Cleaning Pipeline
To maintain 100% numerical consistency between the exploratory notebook and the dashboard, the following steps are enforced during loading:
1. **Handling Missing Values**: Postal Codes in rows with missing values are filled using the modal postal code of their respective `State`.
2. **Date Ingestion & Validation**: Order and Ship dates are parsed. Rows violating chronosequence (i.e. `Ship Date < Order Date`) are dropped.
3. **Data Integrity Checks**: All rows with non-positive sales value (`Sales <= 0`) are filtered.
4. **Column Pruning**: High cardinality metadata columns like `Row ID` and `Country` are removed, and data is sorted chronologically.

---

## 📈 Summary of Key Results

- **Data Consistency**: Verified dataset size is reduced from 9,800 rows to 9,800 validated rows after data cleaning checks.
- **Best Forecasting Model**: **XGBoost (Recursive)** achieved the lowest error metrics, yielding a **13.29% MAPE** on a 3-month holdout test set, outperforming Prophet (**20.3% MAPE**) and SARIMA.
- **Seasonality Patterns**: Sales show a strong **Q4 surge** (driven by holiday shopping spikes) and deep contractions in **Jan-Feb** across all years.
- **Anomaly Detection consensus**: Identified historical sales volatility spikes using Isolation Forest (ML-based) and local Rolling Z-Score (12-Week window), showing dual consensus overlap on major festive shopping weeks.
- **Demand Clustering**: 17 sub-categories were successfully segmented into **High Volume Stable**, **Growing Demand** (Copiers), and **Low Volume Volatile** groups to guide supply chain stocking policies.
