import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Superstore Dashboard", layout="wide")

st.title("📊 Superstore Analytics Dashboard")

# =============================
# 📁 File Upload
# =============================
uploaded_file = st.file_uploader("Upload your dataset (Excel/CSV)", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert numeric columns
    for col in ['Sales', 'Profit', 'Discount', 'Quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(0, inplace=True)

    # =============================
    # 🔎 Filters
    # =============================
    st.sidebar.header("🔎 Filters")

    category = st.sidebar.multiselect("Category", df['Category'].unique(), default=df['Category'].unique())
    region = st.sidebar.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
    segment = st.sidebar.multiselect("Segment", df['Segment'].unique(), default=df['Segment'].unique())

    filtered_df = df[
        (df['Category'].isin(category)) &
        (df['Region'].isin(region)) &
        (df['Segment'].isin(segment))
    ]

    st.write("### 📄 Filtered Data")
    st.dataframe(filtered_df.head())

    # =============================
    # 📊 KPI Metrics
    # =============================
    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
    col2.metric("📈 Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
    col3.metric("📦 Total Quantity", int(filtered_df['Quantity'].sum()))

    # =============================
    # 📊 Interactive Charts
    # =============================

    st.subheader("📊 Profit by Category")
    fig1 = px.bar(filtered_df.groupby('Category')['Profit'].sum().reset_index(),
                  x='Category', y='Profit', color='Category')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Profit by Sub-Category")
    fig2 = px.bar(filtered_df.groupby('Sub-Category')['Profit'].sum().reset_index(),
                  x='Sub-Category', y='Profit', color='Sub-Category')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📉 Discount vs Profit")
    fig3 = px.scatter(filtered_df, x='Discount', y='Profit', color='Category')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🌍 Country-wise Sales Map")
    if 'Country' in filtered_df.columns:
        country_df = filtered_df.groupby('Country')['Sales'].sum().reset_index()

        fig_map = px.choropleth(country_df,
                               locations='Country',
                               locationmode='country names',
                               color='Sales',
                               color_continuous_scale='Blues')
        st.plotly_chart(fig_map, use_container_width=True)

    # =============================
    # 🤖 Machine Learning Model
    # =============================
    st.subheader("🤖 Profit Prediction Model")

    features = ['Sales', 'Discount', 'Quantity']
    if all(col in filtered_df.columns for col in features):

        X = filtered_df[features]
        y = filtered_df['Profit']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        st.success("Model Trained Successfully!")

        # =============================
        # 🔮 Predict New Data
        # =============================
        st.write("### 🔮 Predict Profit")

        sales_input = st.number_input("Enter Sales", value=100.0)
        discount_input = st.number_input("Enter Discount", value=0.1)
        quantity_input = st.number_input("Enter Quantity", value=1)

        if st.button("Predict Profit"):
            prediction = model.predict([[sales_input, discount_input, quantity_input]])
            st.success(f"Predicted Profit: ${prediction[0]:.2f}")

    # =============================
    # 📥 Download Cleaned Data
    # =============================
    st.subheader("📥 Download Cleaned Data")
    st.download_button(
        label="Download CSV",
        data=filtered_df.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload a dataset to begin.")