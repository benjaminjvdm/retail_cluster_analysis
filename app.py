import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetic parameters
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

@st.cache_data
def load_data():
    try:
        product_data = pd.read_csv('Product Data Set.csv', sep='|')
        transactions_data = pd.read_csv('Transaction Data Set.csv', sep='|')
        customer_data = pd.read_csv('Customer Data Set.csv')
        return product_data, transactions_data, customer_data
    except FileNotFoundError as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def clean_data(customer_data):
    try:
        customer_data['INCOME'] = customer_data['INCOME'].str.replace('[\$,]', '', regex=True).astype(float)
        customer_data = customer_data.dropna(subset=['INCOME'])
        return customer_data
    except KeyError as e:
        st.error(f"Missing column in data: {str(e)}")
        st.stop()

@st.cache_data
def merge_data(transactions_data, product_data, customer_data):
    try:
        # Merge transactions with products
        trans_products = transactions_data.merge(
            product_data, 
            how='inner', 
            left_on='PRODUCT NUM', 
            right_on='PRODUCT CODE'
        )
        
        # Clean and calculate prices
        trans_products['UNIT LIST PRICE'] = trans_products['UNIT LIST PRICE'].str.replace('[\$,]', '', regex=True).astype(float)
        trans_products['Total_Price'] = trans_products['QUANTITY PURCHASED'] * trans_products['UNIT LIST PRICE'] * (1 - trans_products['DISCOUNT TAKEN'])
        
        # Create customer-category matrix
        customer_prod_categ = trans_products.groupby(['CUSTOMER NUM', 'PRODUCT CATEGORY']).agg({'Total_Price': 'sum'}).reset_index()
        customer_pivot = customer_prod_categ.pivot(
            index='CUSTOMER NUM', 
            columns='PRODUCT CATEGORY', 
            values='Total_Price'
        ).fillna(0)
        
        # Calculate total spending
        trans_total_spend = trans_products.groupby('CUSTOMER NUM').agg({'Total_Price': 'sum'}).rename(columns={'Total_Price': 'TOTAL SPENT'})
        
        # Merge all customer data
        customer_KPIs = customer_pivot.merge(trans_total_spend, how='inner', left_index=True, right_index=True)
        customer_all_view = customer_data.merge(
            customer_KPIs, 
            how='inner', 
            left_on='CUSTOMERID', 
            right_index=True
        )
        return customer_all_view
    except Exception as e:
        st.error(f"Error merging data: {str(e)}")
        st.stop()

def perform_clustering(customer_all_view):
    st.sidebar.subheader("Clustering Parameters")
    
    # Feature selection
    numeric_cols = customer_all_view.select_dtypes(include=np.number).columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Select features for clustering:",
        options=numeric_cols,
        default=['INCOME', 'TOTAL SPENT']
    )
    
    if not selected_features:
        st.warning("Please select at least one feature for clustering.")
        st.stop()
        
    # Data scaling
    scaler = StandardScaler()
    cluster_input = scaler.fit_transform(customer_all_view[selected_features])
    
    # Cluster number selection with elbow method
    st.sidebar.subheader("Determine Optimal Clusters")
    max_clusters = min(15, len(customer_all_view)-1)
    if st.sidebar.checkbox("Show Elbow Method"):
        distortions = []
        K = range(1, 10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(cluster_input)
            distortions.append(kmeanModel.inertia_)
            
        fig, ax = plt.subplots()
        ax.plot(K, distortions, 'bx-')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Distortion')
        ax.set_title('Elbow Method for Optimal K')
        st.pyplot(fig)
    
    n_clusters = st.sidebar.slider(
        "Number of clusters:", 
        min_value=2, 
        max_value=10, 
        value=4
    )
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(cluster_input)
    
    # Create results dataframe
    segment_DF = customer_all_view[selected_features].copy()
    segment_DF['segment'] = cluster_labels
    
    return segment_DF, kmeans, scaler, selected_features

def plot_clusters(segment_DF, model, scaler, features):
    fig, ax = plt.subplots()
    
    # Inverse transform for original scale
    cluster_centers = scaler.inverse_transform(model.cluster_centers_)
    
    # Create scatter plot
    sns.scatterplot(
        data=segment_DF,
        x=features[0],
        y=features[1],
        hue='segment',
        palette='viridis',
        s=100,
        alpha=0.8,
        ax=ax
    )
    
    # Plot centroids
    ax.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        s=300,
        marker='X',
        c='red',
        edgecolor='black',
        label='Centroids'
    )
    
    ax.set_title(f'Customer Segments: {features[0]} vs {features[1]}')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.legend()
    st.pyplot(fig)

def show_cluster_stats(segment_DF):
    st.subheader("Cluster Statistics")
    
    # Calculate cluster statistics
    cluster_stats = segment_DF.groupby('segment').agg({
        'INCOME': ['mean', 'median', 'std'],
        'TOTAL SPENT': ['mean', 'median', 'std']
    })
    
    # Display statistics
    st.dataframe(cluster_stats.style.format("{:.2f}").background_gradient(cmap='Blues'))
    
    # Add interpretation
    st.markdown("""
    **Interpretation Guide:**
    - **Mean/Median:** Average values for each cluster
    - **Std:** Variation within cluster
    - Compare clusters to identify high/low value groups
    """)

def main():
    st.title('📊 Customer Segmentation for Targeted Marketing')
    st.markdown("""
    This advanced analytics app provides:
    - Customer segmentation using machine learning
    - Interactive data exploration
    - Marketing strategy recommendations
    """)
    
    # Load and process data
    product_data, transactions_data, customer_data = load_data()
    customer_data = clean_data(customer_data)
    customer_all_view = merge_data(transactions_data, product_data, customer_data)
    
    # Perform clustering
    segment_DF, kmeans, scaler, features = perform_clustering(customer_all_view)
    
    # Page navigation
    page = st.sidebar.radio("Navigation", ["Cluster Analysis", "Data Exploration", "Marketing Insights"])
    
    if page == "Cluster Analysis":
        st.header("Cluster Analysis")
        plot_clusters(segment_DF, kmeans, scaler, features)
        show_cluster_stats(segment_DF)
        
    elif page == "Data Exploration":
        st.header("Data Exploration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Income Distribution")
            fig = plt.figure()
            sns.histplot(customer_all_view['INCOME'], kde=True, bins=20)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Spending vs Income")
            fig = plt.figure()
            sns.scatterplot(data=customer_all_view, x='INCOME', y='TOTAL SPENT', alpha=0.6)
            st.pyplot(fig)
            
        st.subheader("Correlation Matrix")
        numeric_df = customer_all_view.select_dtypes(include=np.number)
        corr_matrix = numeric_df.corr()
        fig = plt.figure()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(fig)
        
    elif page == "Marketing Insights":
        st.header("Marketing Strategy Recommendations")
        
        # Generate dynamic insights based on clusters
        avg_income = segment_DF.groupby('segment')['INCOME'].mean()
        high_value_cluster = avg_income.idxmax()
        
        st.markdown(f"""
        ### Key Recommendations:
        - **Cluster {high_value_cluster}** has the highest average income (${avg_income[high_value_cluster]:,.2f})
        - Target this group with premium products and personalized offers
        - Develop loyalty programs for high-spending clusters
        - Create reactivation campaigns for low-engagement clusters
        """)
        
        st.subheader("Cluster Characteristics")
        cluster_desc = segment_DF.groupby('segment').agg({
            'INCOME': 'mean',
            'TOTAL SPENT': 'mean'
        }).reset_index()
        st.dataframe(cluster_desc.style.format({"INCOME": "${:.2f}", "TOTAL SPENT": "${:.2f}"}))

if __name__ == '__main__':
    main()
