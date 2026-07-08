import json

notebook_path = 'analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 1 (Imports)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('import pandas as pd' in line for line in cell['source']):
        # Check if already added
        if not any('LOKY_MAX_CPU_COUNT' in line for line in cell['source']):
            cell['source'].insert(0, "import os\n")
            cell['source'].insert(1, "os.environ[\"LOKY_MAX_CPU_COUNT\"] = \"4\"\n")
            print("Successfully updated imports cell with LOKY_MAX_CPU_COUNT environment variable.")
        break

# 2. Update Cell 21 (XGBoost forecasting)
xgb_new_source = [
    "def create_lag_features(data, lags=3):\n",
    "    df_lag = pd.DataFrame({'Sales': data})\n",
    "    for i in range(1, lags + 1):\n",
    "        df_lag[f'Lag_{i}'] = df_lag['Sales'].shift(i)\n",
    "    df_lag['Rolling_Mean_3'] = df_lag['Sales'].rolling(window=3).mean()\n",
    "    df_lag['Month'] = df_lag.index.month\n",
    "    df_lag['Quarter'] = df_lag.index.quarter\n",
    "    df_lag['Season'] = df_lag['Month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})\n",
    "    df_lag.dropna(inplace=True)\n",
    "    return df_lag\n",
    "\n",
    "train_lag_df = create_lag_features(train)\n",
    "features = ['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Month', 'Quarter', 'Season']\n",
    "target = 'Sales'\n",
    "X_train_xgb = train_lag_df[features]\n",
    "y_train_xgb = train_lag_df[target]\n",
    "\n",
    "xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)\n",
    "xgb_model.fit(X_train_xgb, y_train_xgb)\n",
    "\n",
    "history = list(train.values)\n",
    "predictions = []\n",
    "for i in range(3):\n",
    "    lag_1 = history[-1]\n",
    "    lag_2 = history[-2]\n",
    "    lag_3 = history[-3]\n",
    "    roll_mean = np.mean(history[-3:])\n",
    "    pred_date = test.index[i]\n",
    "    month = pred_date.month\n",
    "    quarter = pred_date.quarter\n",
    "    season = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}[month]\n",
    "    features_row = pd.DataFrame([[lag_1, lag_2, lag_3, roll_mean, month, quarter, season]], columns=features)\n",
    "    pred = xgb_model.predict(features_row)[0]\n",
    "    predictions.append(pred)\n",
    "    history.append(pred)\n",
    "\n",
    "xgb_pred = pd.Series(predictions, index=test.index)\n",
    "xgb_mae = mean_absolute_error(test, xgb_pred)\n",
    "xgb_rmse = np.sqrt(mean_squared_error(test, xgb_pred))\n",
    "xgb_mape = np.mean(np.abs((test - xgb_pred) / test)) * 100\n",
    "print(f\"XGBoost Metrics:\")\n",
    "print(f\"  MAE:  ${xgb_mae:,.2f}\")\n",
    "print(f\"  RMSE: ${xgb_rmse:,.2f}\")\n",
    "print(f\"  MAPE: {xgb_mape:.2f}%\")\n",
    "print(f\"  Forecast: {list(xgb_pred.round(2))}\")"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('xgb.XGBRegressor' in line for line in cell['source']) and any('create_lag_features' in line for line in cell['source']):
        cell['source'] = xgb_new_source
        print("Successfully updated XGBoost forecasting cell to fix data leakage.")
        break

# 3. Update Cell 39 (KMeans Clustering label mapping)
kmeans_new_source = [
    "optimal_k = 3\n",
    "kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)\n",
    "features_df['Cluster'] = kmeans.fit_predict(X_scaled)\n",
    "pca = PCA(n_components=2)\n",
    "X_pca = pca.fit_transform(X_scaled)\n",
    "features_df['PC1'] = X_pca[:, 0]\n",
    "features_df['PC2'] = X_pca[:, 1]\n",
    "cluster_profiles = features_df.groupby('Cluster')[['Total_Sales', 'Growth_Rate', 'Volatility']].mean()\n",
    "print(\"Cluster Profiles:\")\n",
    "print(cluster_profiles)\n",
    "cluster_labels = {}\n",
    "available_clusters = list(range(optimal_k))\n",
    "high_vol_c = cluster_profiles['Total_Sales'].idxmax()\n",
    "cluster_labels[high_vol_c] = 'High Volume, Stable Demand'\n",
    "available_clusters.remove(high_vol_c)\n",
    "remaining_growth = cluster_profiles.loc[available_clusters, 'Growth_Rate']\n",
    "growing_c = remaining_growth.idxmax()\n",
    "cluster_labels[growing_c] = 'Growing Demand'\n",
    "available_clusters.remove(growing_c)\n",
    "for c in available_clusters:\n",
    "    cluster_labels[c] = 'Low Volume, High Volatility'\n",
    "features_df['Cluster_Label'] = features_df['Cluster'].map(cluster_labels)\n",
    "print(f\"\\nCluster Labels: {cluster_labels}\")"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('kmeans = KMeans(n_clusters=optimal_k' in line for line in cell['source']):
        cell['source'] = kmeans_new_source
        print("Successfully updated KMeans clustering cell to resolve label overlaps.")
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook update completed successfully.")
