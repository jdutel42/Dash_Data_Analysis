# ================================
# Import Required Libraries
# =================================
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import dash
from dash import Dash, dcc, html, Input, Output
import dash_daq as daq
import dash_cytoscape as cyto
import community as community_louvain
from scipy.sparse import lil_matrix

# ===================================
# Data Loading and Preprocessing
# ===================================

# ------------------------------
# Part 1: Network Graph and Clustering
# ------------------------------

# Define the file path for sample data
network_raw_file_path = 'sample_data.csv'

# Load the sample data
network_data = pd.read_csv(network_raw_file_path)

# Define the relevant variables for analysis
network_variables = [
    'PhysicalActivities', 'BMI', 'GeneralHealth', 'PhysicalHealthDays',
    'SleepHours', 'MentalHealthDays', 'SmokerStatus', 'ECigaretteUsage',
    'HadCOPD', 'HadHeartAttack', 'HadAsthma', 'AlcoholDrinkers'
]

# Create a copy for analysis
network_analysis_df = network_data[network_variables].copy()

# Handle ordinal variable (GeneralHealth)
health_mapping = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very good': 3,
    'Excellent': 4
}
network_analysis_df['GeneralHealth'] = (
    network_analysis_df['GeneralHealth']
    .str.strip()
    .map(health_mapping)
)

# Handle binary variables
network_binary_vars = ['PhysicalActivities', 'HadCOPD', 'HadHeartAttack', 'HadAsthma', 'AlcoholDrinkers']
for col in network_binary_vars:
    network_analysis_df[col] = network_analysis_df[col].str.strip()
    network_analysis_df[col] = (network_analysis_df[col] == 'Yes').astype(int)

# Handle categorical variables
network_categorical_vars = ['SmokerStatus', 'ECigaretteUsage']
already_network_cat_vars = [col for col in network_categorical_vars if col in network_analysis_df.columns]
if already_network_cat_vars:
    for col in already_network_cat_vars:
        network_analysis_df[col] = network_analysis_df[col].str.strip()
    network_analysis_df = pd.get_dummies(network_analysis_df, columns=already_network_cat_vars, drop_first=True)

# Identify and process numeric variables
network_numeric_vars = ['BMI', 'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'GeneralHealth']
network_scaler = MinMaxScaler()
for col in network_numeric_vars:
    median_value = network_analysis_df[col].median()
    network_analysis_df[col] = network_analysis_df[col].fillna(median_value)
    network_analysis_df[col] = network_scaler.fit_transform(network_analysis_df[[col]])

# Convert boolean columns to integers if any
network_bool_columns = network_analysis_df.select_dtypes(include=['bool']).columns
for col in network_bool_columns:
    network_analysis_df[col] = network_analysis_df[col].astype(int)

# Rename to avoid confusion
network_processed_data = network_analysis_df.copy()

# Define feature categories
network_disease_features = ['HadCOPD', 'HadHeartAttack', 'HadAsthma']
network_behavior_features = [col for col in network_processed_data.columns if col not in network_disease_features]

# Reset index
network_processed_data.reset_index(drop=True, inplace=True)

# Preserve the original feature matrix
network_feature_matrix_original = network_processed_data[network_behavior_features + network_disease_features].copy()

# Adjust feature weights
weight_factor = 3
network_feature_matrix_weighted = network_feature_matrix_original.copy()
network_feature_matrix_weighted[network_disease_features] = network_feature_matrix_weighted[network_disease_features] * weight_factor

# Standardize features
network_scaler_standard = StandardScaler()
network_X_scaled = network_scaler_standard.fit_transform(network_feature_matrix_weighted)

# Compute cosine similarity and build adjacency matrix
K = 10
network_nbrs = NearestNeighbors(n_neighbors=K, metric='cosine').fit(network_X_scaled)
network_distances, network_indices = network_nbrs.kneighbors(network_X_scaled)
network_num_samples = network_X_scaled.shape[0]
network_adjacency_sparse = lil_matrix((network_num_samples, network_num_samples))

for i in range(network_num_samples):
    for j in range(1, K):  # Skip self
        similarity = 1 - network_distances[i][j]
        if similarity >= 0.5:
            network_adjacency_sparse[i, network_indices[i][j]] = 1

network_adjacency_sparse = network_adjacency_sparse.tocsr()

# Build NetworkX graph
G_network = nx.from_scipy_sparse_matrix(network_adjacency_sparse)

# Add node attributes
for node in G_network.nodes():
    G_network.nodes[node]['id'] = node

# Clean the graph
G_network.remove_edges_from(nx.selfloop_edges(G_network))
G_network.remove_nodes_from(list(nx.isolates(G_network)))

# Perform t-SNE on the network graph
network_embedding_method = 't-SNE'
network_n_samples_graph = G_network.number_of_nodes()
if network_n_samples_graph <= 1:
    raise ValueError("Insufficient samples for t-SNE.")
elif network_n_samples_graph < 5:
    network_perplexity = network_n_samples_graph - 1
elif network_n_samples_graph < 30:
    network_perplexity = min(30, network_n_samples_graph - 1)
else:
    network_perplexity = 30

network_tsne_graph = TSNE(n_components=2, random_state=42, perplexity=network_perplexity, learning_rate='auto', n_iter=1000)
network_embeddings_graph = network_tsne_graph.fit_transform(network_X_scaled)
network_embedding_df_graph = pd.DataFrame(network_embeddings_graph, columns=['Dim1', 'Dim2'], index=range(network_num_samples))
network_embedding_df_graph = network_embedding_df_graph.loc[list(G_network.nodes())]

# Perform Louvain community detection
network_partition_graph = community_louvain.best_partition(G_network)
nx.set_node_attributes(G_network, network_partition_graph, 'community')
network_embedding_df_graph['community'] = network_embedding_df_graph.index.map(network_partition_graph)

# Prepare colors for communities
network_num_communities_graph = len(network_embedding_df_graph['community'].unique())
network_color_map_graph = px.colors.qualitative.Plotly
if network_num_communities_graph > len(network_color_map_graph):
    network_color_map_graph = network_color_map_graph * (network_num_communities_graph // len(network_color_map_graph) + 1)
network_embedding_df_graph['color'] = network_embedding_df_graph['community'].apply(lambda x: network_color_map_graph[x])

# Dimensionality reduction for clustering
network_pca = PCA(n_components=2, random_state=42)
network_X_pca = network_pca.fit_transform(network_X_scaled)
network_tsne_clustering = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto', init='pca')
network_X_tsne_clustering = network_tsne_clustering.fit_transform(network_X_scaled)
network_df_pca = pd.DataFrame(network_X_pca, columns=['Dim1', 'Dim2'])
network_df_tsne = pd.DataFrame(network_X_tsne_clustering, columns=['Dim1', 'Dim2'])

# ------------------------------
# Part 2: Health Map and Prediction Model
# ------------------------------

# Define file paths
map_raw_file_path = 'sample_data.csv'  # Assuming 'sample_data.csv' is needed here
map_heart_file_path = 'heart_2020_cleaned.csv'  # Load heart disease data

# Load data
map_data = pd.read_csv(map_raw_file_path)
map_df_heart = pd.read_csv(map_heart_file_path)

# Calculate risk indicators
state_data = map_data.groupby('State').agg(
    avg_physical_health=('PhysicalHealthDays', 'mean'),
    avg_mental_health=('MentalHealthDays', 'mean'),
    smokers=('SmokerStatus', lambda x: (x == 'Current smoker').mean()),
    heart_disease_rate=('HadHeartAttack', lambda x: (x == 'Yes').mean()),
    obesity_rate=('BMI', lambda x: (x > 30).mean()),
    diabetes_rate=('HadDiabetes', lambda x: (x == 'Yes').mean())
).reset_index()

state_data['risk_index'] = (
    state_data['smokers'] * 1.5 +
    state_data['obesity_rate'] * 2 +
    state_data['heart_disease_rate'] * 1.5 +
    state_data['diabetes_rate'] * 1.5 +
    state_data['avg_physical_health'] * 1 +
    state_data['avg_mental_health'] * 1
)

# Prepare clustering
map_cluster_data = state_data.copy()
map_scaler = StandardScaler()
map_features_scaled = map_scaler.fit_transform(map_cluster_data.drop(columns=['State']))

# Initial K-Means clustering
map_kmeans_initial = KMeans(n_clusters=5, random_state=42)
map_cluster_data['cluster'] = map_kmeans_initial.fit_predict(map_features_scaled).astype(str)

# Elbow plot data
map_inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(map_features_scaled)
    map_inertia.append(kmeans.inertia_)

# Load US states GeoJSON
with open("gz_2010_us_040_00_500k.json") as geojson_file:
    map_geojson_data = json.load(geojson_file)

# Prepare prediction model
# Encode categorical columns
predictive_categorical_columns = [
    'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex',
    'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity',
    'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer'
]
map_label_encoders = {col: LabelEncoder() for col in predictive_categorical_columns}

for col in predictive_categorical_columns:
    map_df_heart[col] = map_label_encoders[col].fit_transform(map_df_heart[col])

# Split data
X = map_df_heart.drop(columns=['HeartDisease'])
y = map_df_heart['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------
# Correlation Analysis and Cytoscape Network
# -------------------------

# 1. Remove missing values
data_clean = pd.read_csv(network_raw_file_path).dropna()

# 2. Encode ordinal variables
ordinal_mappings = {
    'GeneralHealth': {
        "Excellent": 1,
        "Very good": 2,
        "Good": 3,
        "Fair": 4,
        "Poor": 5
    },
    'LastCheckupTime': {
        "Within past year (anytime less than 12 months ago)": 1,
        "Within past 2 years (1 year but less than 2 years ago)": 2,
        "Within past 5 years (2 years but less than 5 years ago)": 3,
        "5 years or more ago": 4
    },
    'RemovedTeeth': {
        "1 to 5": 1,
        "6 or more, but not all": 2,
        "All": 3,
        "None": 8
    },
    'HadDiabetes': {
        "Yes": 1,
        "Yes, only during pregnancy (female)": 2,
        "No": 3,
        "No, prediabetes or borderline diabetes": 4
    },
    'AgeCategory': {
        "18-24": 1,
        "25-29": 2,
        "30-34": 3,
        "35-39": 4,
        "40-44": 5,
        "45-49": 6,
        "50-54": 7,
        "55-59": 8,
        "60-64": 9,
        "65-69": 10,
        "70-74": 11,
        "75-79": 12,
        "80+": 13,
    },
    'TetanusLast10Tdap': {
        "Yes, received Tdap": 1,
        "Yes, received tetanus vaccine but not Tdap": 2,
        "Yes, received tetanus vaccine but unsure of type": 3,
        "No, have not received any tetanus vaccine in past 10 years": 4
    },
    'CovidPos': {
        "Yes": 1,
        "No": 2,
        "Positive via home test without healthcare professional": 3
    },
}

for col, mapping in ordinal_mappings.items():
    if col in data_clean.columns:
        data_clean[col] = data_clean[col].map(mapping)

# 3. One-hot encode categorical variables
categorical_cols = [
    'State', 'Sex', 'PhysicalActivities', 'SmokerStatus',
    'ECigaretteUsage', 'RaceEthnicityCategory', 'DeafOrHardOfHearing',
    'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking',
    'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan',
    'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12',
    'PneumoVaxEver', 'HighRiskLastYear'
]
data_encoded = pd.get_dummies(data_clean, columns=categorical_cols, drop_first=True)

# Encode remaining non-numeric columns if any
non_numeric_cols = data_encoded.select_dtypes(include=['object']).columns
binary_mapping = {'Yes': 1, 'No': 0}
for col in non_numeric_cols:
    data_encoded[col] = data_encoded[col].map(binary_mapping)

# Verify all columns are numeric
non_numeric_cols_after = data_encoded.select_dtypes(include=['object']).columns
if len(non_numeric_cols_after) > 0:
    raise ValueError("Some columns are still non-numeric.")

# -------------------------
# Correlation Analysis
# -------------------------

# Identify column types
quantitative_cols = [
    'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours',
    'HeightInMeters', 'WeightInKilograms', 'BMI'
]
ordinal_cols = list(ordinal_mappings.keys())
binary_cols = [col for col in data_encoded.columns if col not in quantitative_cols + ordinal_cols]

# Initialize correlation matrix
all_cols = data_encoded.columns
correlation_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)

# 1. Pearson correlation for quantitative variables
quant_corr = data_encoded[quantitative_cols].corr(method='pearson')
correlation_matrix.update(quant_corr)

# 2. Spearman correlation for ordinal variables
ordinal_corr = data_encoded[ordinal_cols].corr(method='spearman')
correlation_matrix.update(ordinal_corr)

# 3. Spearman correlation between quantitative and ordinal variables
for q_col in quantitative_cols:
    for o_col in ordinal_cols:
        corr_value = data_encoded[[q_col, o_col]].corr(method='spearman').iloc[0, 1]
        correlation_matrix.loc[q_col, o_col] = corr_value
        correlation_matrix.loc[o_col, q_col] = corr_value

# 4. Pearson correlation for binary variables with others
binary_corr = data_encoded[binary_cols + quantitative_cols + ordinal_cols].corr(method='pearson')
correlation_matrix.update(binary_corr)

# Fill NaN with 0
correlation_matrix.fillna(0, inplace=True)

# Create heatmap using Plotly
heatmap_fig = px.imshow(
    correlation_matrix.astype(float),
    labels=dict(x="Variables", y="Variables", color="Correlation"),
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Correlation Heatmap",
    width=1500,
    height=1200
)

# -------------------------
# Create Cytoscape Network Graph
# -------------------------

# Define initial correlation threshold
initial_threshold = 0.3

# Function to create NetworkX graph based on threshold
def create_graph(threshold):
    G = nx.Graph()
    # Add nodes
    for col in correlation_matrix.columns:
        G.add_node(col)
    # Add edges based on threshold
    for i in correlation_matrix.columns:
        for j in correlation_matrix.columns:
            if i != j and abs(correlation_matrix.loc[i, j]) > threshold:
                G.add_edge(i, j, weight=correlation_matrix.loc[i, j])
    return G

G_initial = create_graph(initial_threshold)

# Function to convert NetworkX graph to Cytoscape format
def nx_to_cytoscape(G):
    nodes = []
    for node in G.nodes():
        degree = G.degree(node)
        classes = 'connected' if degree > 0 else 'isolated'
        nodes.append({
            "data": {"id": node, "label": node, "degree": degree},
            "classes": classes  # Assign class based on connectivity
        })
    edges = []
    for edge in G.edges(data=True):
        source, target, data = edge
        edges.append({
            "data": {"source": source, "target": target, "weight": data['weight']}
        })
    return nodes, edges

nodes_cyto, edges_cyto = nx_to_cytoscape(G_initial)

# Define Cytoscape stylesheet
cyto_stylesheet = [
    {
        'selector': 'node.connected',
        'style': {
            'label': 'data(label)',  # Display label for connected nodes
            'width': '60px',
            'height': '60px',
            'background-color': '#0074D9',
            'font-size': '10px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': '120px',
        }
    },
    {
        'selector': 'node.isolated',
        'style': {
            'label': '',
            'width': '60px',
            'height': '60px',
            'background-color': '#0074D9',
            'font-size': '10px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': '120px',
        }
    },
    {
        'selector': 'node:selected',
        'style': {
            'background-color': '#FF4136',
            'width': '80px',
            'height': '80px',
            'font-size': '14px',
            'label': 'data(label)',
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'opacity': 0.7,
            'line-color': '#B3B3B3',
            'width': '2px',
        }
    },
]

# ===============================
# Initialize Dash App
# ===============================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For deployment purposes
app.title = "Comprehensive Health Data Dashboard"

# ================================
# Define Layout
# =================================
app.layout = html.Div([
    html.H1("Comprehensive Health Data Visualization and Analysis Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs([
        # Tab 1: Disease-Behavior Network Graph
        dcc.Tab(label='Disease-Behavior Network Graph', children=[
            html.Div([
                html.H2("Disease and Behavior Feature Relationship Network", style={'textAlign': 'center'}),
                # Network Graph
                dcc.Graph(
                    id='network-relationship-graph',
                    figure={}
                ),
                # Node Information
                html.Div(id='network-node-info', style={
                    'padding': '20px',
                    'border': '1px solid #ccc',
                    'marginTop': '20px'
                }),
                # Description
                html.Div([
                    html.P("Click on a node to view detailed information.")
                ], style={'marginTop': '20px'})
            ], style={'padding': '20px'})
        ]),

        # Tab 2: Clustering and Analysis
        dcc.Tab(label='Clustering and Analysis', children=[
            html.Div([
                html.H2("Clustering and Dimensionality Reduction Analysis", style={'textAlign': 'center'}),
                # Clustering Controls
                html.Div([
                    html.Div([
                        html.Label("Dimensionality Reduction Method:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='network-reduction-dropdown',
                            options=[{'label': m, 'value': m} for m in ['PCA', 't-SNE']],
                            value='PCA',
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                    html.Div([
                        html.Label("Clustering Method:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='network-cluster-dropdown',
                            options=[{'label': m, 'value': m} for m in ['KMeans', 'GMM']],
                            value='KMeans',
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
                ]),
                # Clustering Graph
                dcc.Graph(id='network-cluster-graph'),
                # Clustering Evaluation
                html.Div(id='network-cluster-evaluation', style={
                    'padding': '20px',
                    'border': '1px solid #ccc',
                    'marginTop': '20px'
                })
            ], style={'padding': '20px'})
        ]),

        # Tab 3: Correlation Heatmap
        dcc.Tab(label='Correlation Heatmap', children=[
            html.Div([
                html.H2("Correlation Heatmap", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='heatmap',
                    figure=heatmap_fig
                )
            ], style={'padding': '20px'})
        ]),

        # Tab 4: Correlation Network Graph
        dcc.Tab(label='Correlation Network Graph', children=[
            html.Div([
                html.H2("Correlation Network Graph", style={'textAlign': 'center'}),
                cyto.Cytoscape(
                    id='cytoscape',
                    elements=nodes_cyto + edges_cyto,
                    layout={
                        'name': 'cose',
                        'nodeRepulsion': 8000,
                        'idealEdgeLength': 200,
                        'animate': True,
                        'animationDuration': 1000,
                        'randomize': False,
                        'nodeDimensionsIncludeLabels': True,
                    },
                    style={'width': '100%', 'height': '1000px'},
                    stylesheet=cyto_stylesheet,
                    userZoomingEnabled=True,
                    userPanningEnabled=True,
                )
            ], style={'padding': '20px'}),
            html.Div([
                html.H3("Network Graph Controls"),
                html.Label("Correlation Threshold:"),
                dcc.Slider(
                    id='threshold-slider',
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    value=initial_threshold,
                    marks={i/10: f"{i/10}" for i in range(1, 11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'padding': '20px'}),
            html.Div([
                html.H3("Selected Node Information"),
                html.Div(id='node-data', style={
                    'border': '1px solid #ccc',
                    'padding': '10px',
                    'border-radius': '5px'
                })
            ], style={'padding': '20px'}),
        ]),

        # Tab 5: Cluster Feature Distribution
        dcc.Tab(label='Cluster Feature Distribution', children=[
            html.Div([
                html.H2("Cluster Feature Distribution", style={'textAlign': 'center'}),
                # Feature Distribution Controls
                html.Div([
                    html.Div([
                        html.Label("Clustering Method:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='network-distribution-cluster-dropdown',
                            options=[{'label': m, 'value': m} for m in ['KMeans', 'GMM']],
                            value='KMeans',
                            clearable=False
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                    html.Div([
                        html.Label("Select Features:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='network-distribution-feature-dropdown',
                            options=[{'label': f, 'value': f} for f in network_behavior_features + network_disease_features],
                            value=[network_behavior_features[0]],
                            multi=True
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
                ]),
                # Distribution Graph
                dcc.Graph(id='network-distribution-graph')
            ], style={'padding': '20px'})
        ]),

        # Tab 6: Health Map
        dcc.Tab(label='Health Map', children=[
            html.Div([
                html.H2("Interactive Health and Lifestyle Map by State", style={'textAlign': 'center'}),
                # Data Selector Dropdown
                dcc.Dropdown(
                    id='map-data-selector',
                    options=[
                        {'label': 'Average Physical Health', 'value': 'avg_physical_health'},
                        {'label': 'Average Mental Health', 'value': 'avg_mental_health'},
                        {'label': 'Smoking Rate', 'value': 'smokers'},
                        {'label': 'Heart Disease Rate', 'value': 'heart_disease_rate'},
                        {'label': 'Obesity Rate', 'value': 'obesity_rate'},
                        {'label': 'Diabetes Rate', 'value': 'diabetes_rate'},
                        {'label': 'Risk Index', 'value': 'risk_index'}
                    ],
                    value='avg_physical_health',
                    style={'width': '50%', 'margin': 'auto'}
                ),
                # US Map
                html.Div(
                    dcc.Graph(id='us-health-map'),
                    style={
                        'width': '95%',
                        'height': '85vh',
                        'margin': 'auto',
                        'padding': '10px',
                        'backgroundColor': '#ffffff',
                        'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.1)',
                        'borderRadius': '10px'
                    }
                ),
                # Elbow Plot and Cluster Map
                html.Div([
                    # Elbow Plot
                    html.Div([
                        dcc.Graph(
                            id='map-elbow-plot',
                            figure={
                                'data': [{'x': list(range(2, 11)), 'y': map_inertia, 'type': 'line', 'name': 'Inertia'}],
                                'layout': {
                                    'title': 'Elbow Method (K vs. Inertia)',
                                    'xaxis': {'title': 'Number of Clusters'},
                                    'yaxis': {'title': 'Inertia'},
                                    'showlegend': False
                                }
                            }
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                    # Cluster Map
                    html.Div([
                        dcc.Dropdown(
                            id='map-num-clusters',
                            options=[{'label': f'{k} Clusters', 'value': k} for k in range(2, 11)],
                            value=5,
                            placeholder="Select Number of Clusters",
                            style={'width': '100%', 'margin': 'auto', 'marginTop': '20px'}
                        ),
                        dcc.Graph(id='map-cluster-map')
                    ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'width':'100%'}),
                # Footer
                html.Footer(
                    "Data sourced from your CSV files | Created with Dash and Plotly",
                    style={
                        'textAlign': 'center',
                        'padding': '10px',
                        'backgroundColor': '#f5f5f5',
                        'color': '#666',
                        'fontSize': '12px'
                    }
                )
            ], style={'padding': '20px'})
        ]),

        # Tab 7: Cardiovascular Risk Prediction
        dcc.Tab(label='Cardiovascular Risk Prediction', children=[
            html.Div([
                html.H2("Cardiovascular Risk Prediction", style={'textAlign': 'center'}),
                # Input Controls
                html.Div([
                    html.Div([
                        html.Label("Smoking (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-smoking', on=False, color="#00cf09")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Alcohol Drinking (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-alcohol-drinking', on=False, color="#00cf09")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Stroke (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-stroke', on=False, color="#00cf09")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Difficulty Walking (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-diff-walking', on=False, color="#00cf09")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Diabetic (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-diabetic', on=False, color="#00cf09")
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Physical Activity (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-physical-activity', on=True, color="#00cf09"),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Asthma (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-asthma', on=False, color="#00cf09"),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Kidney Disease (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-kidney-disease', on=False, color="#00cf09"),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Skin Cancer (Yes/No)", style={'marginRight': '10px'}),
                        daq.BooleanSwitch(id='predict-skin-cancer', on=False, color="#00cf09"),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Sex (Male/Female)", style={'marginRight': '10px'}),
                        dcc.RadioItems(
                            id='predict-sex',
                            options=[
                                {
                                    "label": html.Div([
                                        html.Img(src="/assets/masc.png", height=30),
                                        html.Span(" Male", style={'fontSize': 15, 'paddingLeft': 10, 'color': 'blue'})
                                    ]),
                                    "value": 1
                                },
                                {
                                    "label": html.Div([
                                        html.Img(src="/assets/fem.png", height=30),
                                        html.Span(" Female", style={'fontSize': 15, 'paddingLeft': 10, 'color': 'pink'})
                                    ]),
                                    "value": 0
                                }
                            ],
                            value=0,
                            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Race"),
                        dcc.Dropdown(
                            id='predict-race',
                            options=[
                                {'label': 'American Indian/Alaska Native', 'value': 0},
                                {'label': 'Asian', 'value': 1},
                                {'label': 'Black', 'value': 2},
                                {'label': 'Hispanic', 'value': 3},
                                {'label': 'Other', 'value': 4},
                                {'label': 'White', 'value': 5}
                            ],
                            value=5,
                            style={
                                'color': 'black',
                                'backgroundColor': 'white',
                            }
                        )
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Label("General Health Status (Excellent/Good/Fair/Poor)", style={'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='predict-gen-health',
                            options=[
                                {'label': 'Excellent', 'value': 0},
                                {'label': 'Good', 'value': 1},
                                {'label': 'Fair', 'value': 2},
                                {'label': 'Poor', 'value': 3},
                                {'label': 'Very Poor', 'value': 4}
                            ],
                            value=1,
                            style={
                                'color': 'black',
                                'backgroundColor': 'white',
                            }
                        )
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Age Category"),
                        dcc.Dropdown(
                            id='predict-age-category',
                            options=[
                                {'label': '18-24', 'value': 0},
                                {'label': '25-29', 'value': 1},
                                {'label': '30-34', 'value': 2},
                                {'label': '35-39', 'value': 3},
                                {'label': '40-44', 'value': 4},
                                {'label': '45-49', 'value': 5},
                                {'label': '50-54', 'value': 6},
                                {'label': '55-59', 'value': 7},
                                {'label': '60-64', 'value': 8},
                                {'label': '65-69', 'value': 9},
                                {'label': '70-74', 'value': 10},
                                {'label': '75-79', 'value': 11},
                                {'label': '80+', 'value': 12},
                            ],
                            value=0,
                            style={
                                'color': 'black',
                                'backgroundColor': 'white',
                            }
                        )
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Physical Health (Last 30 Days)"),
                        dcc.Slider(
                            id='predict-physical-health',
                            min=0,
                            max=30,
                            step=1,
                            value=0,
                            marks={i: str(i) for i in range(0, 31, 5)}
                        )
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Mental Health (Last 30 Days)"),
                        dcc.Slider(
                            id='predict-mental-health',
                            min=0,
                            max=30,
                            step=1,
                            value=0,
                            marks={i: str(i) for i in range(0, 31, 5)}
                        )
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Label("BMI (Body Mass Index)"),
                        dcc.Input(id='predict-bmi', type='number', value=25),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Button('Predict', id='predict-submit-button', n_clicks=0),
                    ], style={'marginTop': '20px'}),
                    html.H3("Result:", style={'marginTop': '20px'}),
                    html.Div([
                        dcc.Graph(id='predict-risk-gauge', style={'backgroundColor': '#2c3e50','width': '50%', 'margin': 'auto'}),
                    ])
                ], style={
                    'backgroundColor': '#2c3e50',
                    'color': 'white',
                    'padding': '20px'
                })
            ], style={'padding': '20px'})
        ]),
    ])
])

# ====================================
# Define Callbacks
# ====================================

# ------------------------------
# Network Graph Callback
# ------------------------------
@app.callback(
    Output('network-relationship-graph', 'figure'),
    Input('network-relationship-graph', 'id')  # Trigger on initialization
)
def update_network_graph(_):
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G_network.edges():
        x0, y0 = network_embedding_df_graph.loc[edge[0], 'Dim1'], network_embedding_df_graph.loc[edge[0], 'Dim2']
        x1, y1 = network_embedding_df_graph.loc[edge[1], 'Dim1'], network_embedding_df_graph.loc[edge[1], 'Dim2']
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.3, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_trace = go.Scatter(
        x=network_embedding_df_graph['Dim1'],
        y=network_embedding_df_graph['Dim2'],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=network_embedding_df_graph['community'],
            colorscale='Viridis',
            size=6,
            colorbar=dict(
                title='Community',
                xanchor='left',
                titleside='right'
            ),
            line_width=1
        ),
        text=network_embedding_df_graph.index.astype(str)
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Disease and Behavior Feature Relationship Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    return fig

# Callback to display node information in Disease-Behavior Network Graph
@app.callback(
    Output('network-node-info', 'children'),
    Input('network-relationship-graph', 'clickData')
)
def display_network_node_info(clickData):
    if clickData is None:
        return html.Div([
            html.H4("Node Information"),
            html.P("Click on a node to view detailed information.")
        ])
    # Extract node name
    point = clickData['points'][0]
    node_name = point['text']
    try:
        node_index = int(node_name)
    except ValueError:
        return html.Div([
            html.H4("Node Information"),
            html.P("Unable to parse the clicked node.")
        ])
    if node_index not in network_embedding_df_graph.index:
        return html.Div([
            html.H4("Node Information"),
            html.P("Clicked node does not exist.")
        ])
    node_data = network_embedding_df_graph.loc[node_index]
    info = [
        html.H4(f"Sample ID: {node_index}"),
        html.P(f"Community: {node_data['community']}"),
        html.P(f"Coordinates: ({node_data['Dim1']:.2f}, {node_data['Dim2']:.2f})")
    ]
    original_values = network_feature_matrix_original.iloc[node_index].values
    original_df = pd.Series(original_values, index=network_behavior_features + network_disease_features)
    info.append(html.H5("Original Feature Values:"))
    table_rows = [
        html.Tr([html.Th("Feature"), html.Th("Value")])
    ] + [
        html.Tr([html.Td(feat), html.Td(f"{val}")]) for feat, val in original_df.items()
    ]
    info.append(html.Table(table_rows, style={
        'width': '50%',
        'border': '1px solid #ccc',
        'borderCollapse': 'collapse',
        'marginTop': '10px'
    }, className='table'))
    return html.Div(info)

# --------------------------------
# Clustering and Analysis Callbacks
# --------------------------------

def network_perform_kmeans(data_to_cluster, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data_to_cluster)
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(data_to_cluster, labels)
    return labels, inertia, silhouette_avg

def network_perform_gmm(data_to_cluster, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data_to_cluster)
    labels = gmm.predict(data_to_cluster)
    silhouette_avg = silhouette_score(data_to_cluster, labels)
    return labels, silhouette_avg

@app.callback(
    [Output('network-cluster-graph', 'figure'),
     Output('network-cluster-evaluation', 'children')],
    [Input('network-reduction-dropdown', 'value'),
     Input('network-cluster-dropdown', 'value')]
)
def update_network_cluster_graph_and_evaluation(reduction_method, cluster_method):
    # Select dimensionality reduction method
    if reduction_method == 'PCA':
        data_reduced = network_X_pca
        df_plot = network_df_pca.copy()
    else:
        data_reduced = network_X_tsne_clustering
        df_plot = network_df_tsne.copy()

    fig = go.Figure()
    fig_eval = html.Div()

    if cluster_method == 'KMeans':
        inertia_list = []
        silhouette_scores = []
        K_values = range(2, 11)
        for k in K_values:
            labels, inertia, silhouette_avg = network_perform_kmeans(data_reduced, k)
            inertia_list.append(inertia)
            silhouette_scores.append(silhouette_avg)
        best_k_index = np.argmax(silhouette_scores)
        best_k = K_values[best_k_index]
        labels, inertia, silhouette_avg = network_perform_kmeans(data_reduced, best_k)
        cluster_label_col = f'KMeans_Cluster'
        df_plot[cluster_label_col] = labels.astype(str)
        df_full = pd.concat([df_plot.reset_index(drop=True), network_processed_data.reset_index(drop=True)], axis=1)
        hover_columns = network_behavior_features + network_disease_features
        fig = px.scatter(
            df_full,
            x='Dim1',
            y='Dim2',
            color=cluster_label_col,
            title=f"{reduction_method} and KMeans (K={best_k})",
            labels={'color': cluster_label_col},
            hover_data=hover_columns
        )
        fig_eval = html.Div([
            html.H3("K-Means Elbow Method (Inertia vs. K)", style={'textAlign': 'center'}),
            dcc.Graph(
                figure=px.line(
                    x=list(K_values),
                    y=inertia_list,
                    title="Elbow Method (Inertia vs. K)",
                    labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'}
                )
            ),
            html.P(f"Optimal K Value: {best_k}", style={'textAlign': 'center'})
        ])

    elif cluster_method == 'GMM':
        gmm_scores = []
        n_components_range = range(2, 11)
        for n in n_components_range:
            labels, silhouette_avg = network_perform_gmm(data_reduced, n)
            gmm_scores.append(silhouette_avg)
        best_n_index = np.argmax(gmm_scores)
        best_n = n_components_range[best_n_index]
        labels, silhouette_avg = network_perform_gmm(data_reduced, best_n)
        cluster_label_col = f'GMM_Cluster'
        df_plot[cluster_label_col] = labels.astype(str)
        df_full = pd.concat([df_plot.reset_index(drop=True), network_processed_data.reset_index(drop=True)], axis=1)
        hover_columns = network_behavior_features + network_disease_features
        fig = px.scatter(
            df_full,
            x='Dim1',
            y='Dim2',
            color=cluster_label_col,
            title=f"{reduction_method} and GMM (Components={best_n})",
            labels={'color': cluster_label_col},
            hover_data=hover_columns
        )
        fig_eval = html.Div([
            html.H3("GMM Silhouette Scores (Silhouette Score vs. Number of Components)", style={'textAlign': 'center'}),
            dcc.Graph(
                figure=px.line(
                    x=list(n_components_range),
                    y=gmm_scores,
                    title="GMM Silhouette Scores vs. Number of Components",
                    labels={'x': 'Number of Components', 'y': 'Silhouette Score'}
                )
            ),
            html.P(f"Optimal Number of Components: {best_n}, Highest Silhouette Score: {silhouette_avg:.4f}", style={'textAlign': 'center'})
        ])

    return fig, fig_eval

# Callback to update cluster feature distribution
@app.callback(
    Output('network-distribution-graph', 'figure'),
    [Input('network-distribution-cluster-dropdown', 'value'),
     Input('network-distribution-feature-dropdown', 'value'),
     Input('network-cluster-graph', 'figure')]
)
def update_network_distribution_graph(cluster_method, selected_features, _):
    if not selected_features:
        return go.Figure()

    # Retrieve optimal K or N based on clustering method
    if cluster_method == 'KMeans':
        # Assuming best K value is 5
        best_k = 5
        labels, _, _ = network_perform_kmeans(network_X_pca, best_k)
        cluster_label = labels.astype(str)
    elif cluster_method == 'GMM':
        # Assuming best N value is 5
        best_n = 5
        labels, _ = network_perform_gmm(network_X_pca, best_n)
        cluster_label = labels.astype(str)
    else:
        cluster_label = 'Undefined'

    df = network_processed_data.copy()
    df['Cluster'] = cluster_label

    # Ensure selected_features is a list
    if isinstance(selected_features, list):
        features = selected_features
    else:
        features = [selected_features]

    # Check if features exist
    features = [f for f in features if f in df.columns]
    if not features:
        return go.Figure()

    fig = go.Figure()
    for feature in features:
        fig.add_trace(go.Box(
            y=df[feature],
            x=df['Cluster'],
            name=feature,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))

    fig.update_layout(
        title=f"{cluster_method} Cluster Feature Distribution",
        xaxis_title="Cluster",
        yaxis_title="Feature Value",
        boxmode='group'
    )

    return fig

# --------------------------------
# Health Map Callbacks
# --------------------------------

@app.callback(
    Output('us-health-map', 'figure'),
    [Input('map-data-selector', 'value')]
)
def update_us_health_map(selected_data):
    fig = px.choropleth(
        state_data,
        geojson=map_geojson_data,
        locations='State',
        featureidkey="properties.NAME",
        color=selected_data,
        title=f"Selected Data: {selected_data}",
        color_continuous_scale="Viridis"
    )
    fig.update_geos(
        scope="usa",
        projection={"type": "albers usa"},
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="lightgray",
        showlakes=True,
        lakecolor="Blue",
        showocean=True,
        oceancolor="LightBlue"
    )
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title_x=0.5,
        font=dict(family="Arial", size=14),
        dragmode="zoom"
    )
    return fig

# Callback to update cluster map
@app.callback(
    Output('map-cluster-map', 'figure'),
    Input('map-num-clusters', 'value')
)
def update_map_cluster_map(k):
    if k is None:
        k = 5  # Default value
    kmeans = KMeans(n_clusters=k, random_state=42)
    map_cluster_data['cluster'] = kmeans.fit_predict(map_features_scaled).astype(str)
    fig = px.choropleth(
        map_cluster_data,
        geojson=map_geojson_data,
        locations='State',
        featureidkey="properties.NAME",
        color='cluster',
        title=f'State Clustering (k={k})',
        color_discrete_sequence=px.colors.qualitative.Set1,
        scope="usa"
    )
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title_x=0.5,
        font=dict(family="Arial", size=14),
        dragmode="zoom"
    )
    return fig

# --------------------------------
# Cardiovascular Risk Prediction Callback
# --------------------------------

@app.callback(
    Output('predict-risk-gauge', 'figure'),
    [
        Input("predict-submit-button", "n_clicks"),
        Input("predict-bmi", "value"),
        Input("predict-smoking", "on"),
        Input("predict-alcohol-drinking", "on"),
        Input("predict-stroke", "on"),
        Input("predict-diff-walking", "on"),
        Input("predict-diabetic", "on"),
        Input("predict-physical-activity", "on"),
        Input("predict-asthma", "on"),
        Input("predict-kidney-disease", "on"),
        Input("predict-skin-cancer", "on"),
        Input("predict-sex", "value"),
        Input("predict-age-category", "value"),
        Input("predict-gen-health", "value"),
        Input("predict-race", "value"),
        Input("predict-physical-health", "value"),
        Input("predict-mental-health", "value")
    ]
)
def update_predict_risk_gauge(
    n_clicks,
    bmi,
    smoking,
    alcohol_drinking,
    stroke,
    diff_walking,
    diabetic,
    physical_activity,
    asthma,
    kidney_disease,
    skin_cancer,
    sex,
    age_category,
    gen_health,
    race,
    physical_health,
    mental_health
):
    if n_clicks == 0:
        return go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            title={'text': "Cardiovascular Risk"},
            gauge={
                'axis': {'range': [0,1]},
                'steps': [
                    {'range': [0,0.3], 'color': "green"},
                    {'range': [0.3,0.7], 'color': "orange"},
                    {'range': [0.7,1], 'color': "red"},
                ],
            }
        ))

    # Prepare input data for prediction
    input_data = pd.DataFrame([{
        "BMI": bmi,
        "Smoking": 1 if smoking else 0,
        "AlcoholDrinking": 1 if alcohol_drinking else 0,
        "Stroke": 1 if stroke else 0,
        "DiffWalking": 1 if diff_walking else 0,
        "Sex": sex,
        "AgeCategory": age_category,
        "PhysicalActivity": 1 if physical_activity else 0,
        "GenHealth": gen_health,
        "Asthma": 1 if asthma else 0,
        "KidneyDisease": 1 if kidney_disease else 0,
        "SkinCancer": 1 if skin_cancer else 0,
        "Race": race,
        "Diabetic": 1 if diabetic else 0,
        "PhysicalHealth": physical_health,
        "MentalHealth": mental_health
    }])

    # Add missing columns with default values if any
    for col in X_train.columns:
        if col not in input_data:
            input_data[col] = 0

    # Reorder columns to match training data
    input_data = input_data[X_train.columns]

    # Predict risk probability
    risk_probability = model.predict_proba(input_data)[0][1]

    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability,
        title={
            'text': "Cardiovascular Risk",
            'font': {'color': "white"}
        },
        number={
            'font': {'color': 'white'}
        },
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, 0.3], 'color': 'green'},
                {'range': [0.3, 0.7], 'color': 'orange'},
                {'range': [0.7, 1], 'color': 'red'},
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': risk_probability
            }
        }
    ))
    fig.update_layout(
        margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
        paper_bgcolor="#2c3e50",
    )
    return fig

# -------------------------
# Cytoscape Network Graph Callbacks
# -------------------------
@app.callback(
    Output('cytoscape', 'elements'),
    Input('threshold-slider', 'value')
)
def update_network(threshold_value):
    # Recreate the graph based on the new threshold
    G_updated = create_graph(threshold_value)
    nodes_updated, edges_updated = nx_to_cytoscape(G_updated)
    return nodes_updated + edges_updated

@app.callback(
    Output('node-data', 'children'),
    [Input('cytoscape', 'tapNodeData'),
     Input('cytoscape', 'elements')]
)
def display_node_data(data, elements):
    if data:
        node = data['label']
        # Find all connected edges
        connected_nodes = []
        for elem in elements:
            if 'source' in elem['data']:
                if elem['data']['source'] == node:
                    connected_node = elem['data']['target']
                    correlation = correlation_matrix.loc[node, connected_node]
                    connected_nodes.append((connected_node, correlation))
                elif elem['data']['target'] == node:
                    connected_node = elem['data']['source']
                    correlation = correlation_matrix.loc[node, connected_node]
                    connected_nodes.append((connected_node, correlation))
        
        if not connected_nodes:
            return html.Div([
                html.P(f"Node: {node}"),
                html.P("No connected nodes.")
            ])
        
        # Create a table of connected nodes and their correlations
        table_rows = [
            html.Tr([html.Th("Connected Node"), html.Th("Correlation")])
        ] + [
            html.Tr([html.Td(conn_node), html.Td(f"{corr:.2f}")]) for conn_node, corr in connected_nodes
        ]
        
        return html.Div([
            html.P(f"Node: {node}"),
            html.H5("Connected Nodes and Correlations:"),
            html.Table(table_rows, style={
                'width': '50%',
                'border': '1px solid #ccc',
                'borderCollapse': 'collapse',
                'marginTop': '10px'
            }, className='table')
        ])
    return "Click on a node to view detailed information."

# ===================================
# Run Dash App
# ====================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8099)
