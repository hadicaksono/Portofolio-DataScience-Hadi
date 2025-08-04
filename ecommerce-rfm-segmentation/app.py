import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation RFM", layout="wide")

st.title("📦 Customer Segmentation with RFM Analysis")

uploaded_file = st.file_uploader("Upload E-commerce CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = pca_result[:, 0]
    rfm['PCA2'] = pca_result[:, 1]

    label_map = {
        0: 'Loyal Big Spenders',
        1: 'At Risk / Churn',
        2: 'Recent VIP',
        3: 'Potential Loyalists'
    }
    rfm['Cluster_Label'] = rfm['Cluster'].map(label_map)

    st.subheader("📊 Cluster Summary")
    summary = rfm.groupby('Cluster_Label')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    st.dataframe(summary)

    st.subheader("🧭 Cluster Visualization (PCA)")
    fig, ax = plt.subplots()
    for label in rfm['Cluster_Label'].unique():
        cluster = rfm[rfm['Cluster_Label'] == label]
        ax.scatter(cluster['PCA1'], cluster['PCA2'], label=label, alpha=0.6)
    ax.legend()
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Customer Segments (2D Projection)")
    st.pyplot(fig)

    st.subheader("🧾 Customer Detail Data")
    st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster_Label']])
