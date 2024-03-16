import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data():
    product_data = pd.read_csv('Product Data Set.csv',sep='|')
    transactions_data = pd.read_csv('Transaction Data Set.csv',sep='|')
    customer_data=pd.read_csv('Customer Data Set.csv')
    return product_data, transactions_data, customer_data

def clean_data(customer_data):
    customer_data['INCOME']=customer_data['INCOME'].map(lambda x : x.replace('$',''))
    customer_data['INCOME']=customer_data['INCOME'].map(lambda x : int(x.replace(',','')))
    return customer_data

def merge_data(transactions_data, product_data, customer_data):
    trans_products = transactions_data.merge(product_data, how='inner', left_on='PRODUCT NUM', right_on='PRODUCT CODE')
    trans_products['UNIT LIST PRICE'] = trans_products['UNIT LIST PRICE'].map(lambda x: float(x.replace('$','')))
    trans_products['Total_Price'] = trans_products['QUANTITY PURCHASED'] * trans_products['UNIT LIST PRICE'] * (1 - trans_products['DISCOUNT TAKEN'])
    customer_prod_categ = trans_products.groupby(['CUSTOMER NUM', 'PRODUCT CATEGORY']).agg({'Total_Price': 'sum'})
    customer_prod_categ = customer_prod_categ.reset_index()
    customer_pivot = customer_prod_categ.pivot(index='CUSTOMER NUM', columns='PRODUCT CATEGORY', values='Total_Price')
    trans_total_spend = trans_products.groupby('CUSTOMER NUM').agg({'Total_Price': 'sum'}).rename(columns={'Total_Price': 'TOTAL SPENT'})
    customer_KPIs = customer_pivot.merge(trans_total_spend, how='inner', left_index=True, right_index=True)
    customer_KPIs = customer_KPIs.fillna(0)
    customer_all_view = customer_data.merge(customer_KPIs, how='inner', left_on='CUSTOMERID', right_index=True)
    return customer_all_view

def perform_clustering(customer_all_view):
    # Allow user input for features to cluster on
    selected_features = ['INCOME', 'TOTAL SPENT'] 

    cluster_input = customer_all_view[selected_features]

    # Allow user input for number of clusters
    n_clusters = st.sidebar.number_input("Number of clusters:", min_value=2, max_value=10, value=4)

    # Cluster the data
    Kmeans_model = KMeans(n_clusters=n_clusters)
    cluster_output = Kmeans_model.fit_predict(cluster_input)
    segment_DF = pd.concat([cluster_input, pd.DataFrame(cluster_output, columns=['segment'])], axis=1)

    return segment_DF, Kmeans_model


def plot_clusters(segment_DF, Kmeans_model):
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['purple', 'blue', 'green', 'cyan', 'red', 'magenta', 'yellow', 'black', 'orange', 'brown']  # Expand this list if needed

    for i in range(Kmeans_model.n_clusters):  # Iterate over the number of clusters
        cluster_data = segment_DF[segment_DF.segment == i]
        ax.scatter(cluster_data['INCOME'], cluster_data['TOTAL SPENT'], s=50, c=colors[i], label=f'Cluster {i+1}')

    ax.scatter(Kmeans_model.cluster_centers_[:,0], Kmeans_model.cluster_centers_[:,1], s=200, marker='s', c='red', alpha=0.7, label='Centroids')

    ax.set_title('Customer segments using K-means')  # Update the title dynamically if you'd like
    ax.set_xlabel('Income')
    ax.set_ylabel('Total Spend')
    ax.legend()

    st.pyplot(fig)


def main():
    st.title('Customer Segmentation for Targeted Marketing')
    st.write("This app aids in understanding customer behavior to tailor marketing campaigns and increase revenue.")

    product_data, transactions_data, customer_data = load_data()
    customer_data = clean_data(customer_data)
    customer_all_view = merge_data(transactions_data, product_data, customer_data)
    segment_DF, Kmeans_model = perform_clustering(customer_all_view)

    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Data Analysis"])

    if page == "Homepage":
        plot_clusters(segment_DF, Kmeans_model)
    elif page == "Data Analysis":
        # Exploratory Visualizations
        st.markdown("## Exploratory Visualizations")
        fig, ax = plt.subplots()
        ax.hist(customer_all_view['INCOME'], bins=10)
        ax.set_title('Income Distribution')
        st.pyplot(fig)

        # Descriptive Statistics
        st.markdown("## Descriptive Statistics")
        st.write(customer_all_view.describe())

        # Insightful Commentary
        st.markdown("## Insightful Commentary")
        st.write("""
        Here you can write your insightful commentary. For example:
        - The average income of our customers is $85,792.482.
        - There is a positive correlation between income and total spend.
        """)

if __name__ == '__main__':
    main()
