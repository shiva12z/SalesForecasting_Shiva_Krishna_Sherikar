# 📊 End-to-End Sales Forecasting & Demand Intelligence System

An intelligent sales forecasting, anomaly detection, and product demand segmentation pipeline built on 4 years (2015–2018) of historical Superstore retail transaction data and global video game industry sales benchmarks.

---

## 🏗️ Codebase Structure

- **`analysis.ipynb`**: R&D Jupyter notebook covering EDA, stationarity testing, model training/backtesting (SARIMA, Prophet, XGBoost), anomaly detection comparison (Isolation Forest vs. Z-Score), and KMeans clustering.
- **`app.py`**: Streamlit interactive dashboard showcasing sales trends, interactive forecasts (Prophet with backtesting metrics), anomaly details, and product demand cluster profiles.
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
Start the live interactive forecasting dashboard:

```bash
streamlit run app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 📈 Summary of Key Results

- **Best Forecasting Model**: **XGBoost (Recursive)** achieved the lowest error metrics, yielding a **13.29% MAPE** on a 3-month holdout test set.
- **Seasonality Patterns**: Sales show a strong **Q4 surge** (driven by holiday shopping spikes) and deep contractions in **Jan-Feb** across all years.
- **Demand Clustering**: 17 sub-categories were successfully segmented into **High Volume Stable**, **Growing Demand** (Copiers), and **Low Volume Volatile** groups to guide supply chain stocking policies.
