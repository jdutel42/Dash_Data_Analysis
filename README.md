# Comprehensive Health Data Visualization and Analysis Dashboard

This project presents an interactive dashboard developed using Plotly Dash for the visualization and analysis of complex health-related datasets. The application integrates multiple analytical layers, including clustering, dimensionality reduction, network analysis, and predictive modeling, with the aim of extracting interpretable insights from large-scale epidemiological datasets.

---

## Features

- **Disease–Behavior Network Graph**: Constructs a feature-similarity graph based on cosine proximity and community detection (Louvain algorithm) to illustrate interactions between behavioral patterns and disease prevalence.
- **Dimensionality Reduction and Clustering**: Implements PCA and t-SNE for visualization, coupled with KMeans and Gaussian Mixture Models (GMM) for clustering analysis. The app provides performance metrics (inertia, silhouette score) to assist in optimal model selection.
- **Feature Distribution Analysis**: Allows comparative analysis of feature distributions across clusters, facilitating interpretability of latent subgroups.
- **Correlation Heatmap and Network**: Computes Pearson and Spearman correlations across variables. Outputs are visualized as a heatmap and as a dynamic Cytoscape network with user-adjustable thresholds.
- **Geographical Health Map**: Aggregates health metrics by U.S. state and applies KMeans clustering to reveal spatial health patterns. Includes choropleth mapping and elbow method visualization.
- **Cardiovascular Risk Prediction**: Uses a logistic regression model trained on cleaned heart disease data to predict individual risk based on user-inputted health parameters.

---

## Data

The application utilizes two main datasets:
- `sample_data.csv`: A general health and behavior dataset with multiple features.
- `heart_2020_cleaned.csv`: A cleaned dataset focused on cardiovascular health for predictive modeling.

---

## Installation

Ensure the following dependencies are installed:

```bash
pip install pandas numpy scikit-learn plotly dash dash_daq dash-cytoscape networkx python-louvain
```

---

## Running the Application

Place the following files in the root directory:
- sample_data.csv
- heart_2020_cleaned.csv
- gz_2010_us_040_00_500k.json

Then run:

```bash
python Dash_app.py
```

The dashboard will be accessible at [http://localhost:8099](http://localhost:8099).

---

## Intended Use

This application was developed as part of an academic coursework in data science applied to public health. It aims to serve as a pedagogical tool and proof of concept for multidimensional health data exploration.
Author

Developed by Jordan Dutel, Ariane Paradan and Pengjun Li, as part of "UE Analyse de données" at Claude Bernard Lyon 1 University.


