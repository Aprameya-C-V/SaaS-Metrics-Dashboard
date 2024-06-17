import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from fpdf import FPDF
from sklearn.linear_model import LinearRegression

# Title of the dashboard
st.title("SaaS Metrics Dashboard")

# Generate Synthetic Data
def generate_synthetic_data():
    np.random.seed(42)
    n_customers = 500
    n_months = 24
    plan_types = ['Basic', 'Standard', 'Premium']
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa']

    # Generate customer data
    customer_ids = np.arange(1, n_customers + 1)
    start_dates = np.random.choice(pd.date_range(start="2022-01-01", end="2023-01-01"), n_customers)
    start_dates = pd.to_datetime(start_dates)  # Ensure start_dates are datetime objects
    end_dates = [start + timedelta(days=int(np.random.randint(30, 365))) for start in start_dates]
    plans = np.random.choice(plan_types, n_customers, p=[0.4, 0.4, 0.2])
    regions = np.random.choice(regions, n_customers)

    # Generate subscription data
    amounts = np.random.randint(20, 500, n_customers)
    statuses = np.random.choice(["active", "canceled"], n_customers, p=[0.8, 0.2])

    data = {
        'Customer ID': customer_ids,
        'Start Date': start_dates,
        'End Date': end_dates,
        'Amount': amounts,
        'Status': statuses,
        'Plan': plans,
        'Region': regions
    }

    return pd.DataFrame(data)

# Sidebar for custom data input
st.sidebar.header("Custom Data Input")
custom_data = st.sidebar.file_uploader("Upload your data CSV file", type=["csv"])
if custom_data is not None:
    data = pd.read_csv(custom_data)
else:
    data = generate_synthetic_data()

# Convert timestamps to datetime
data['Start Date'] = pd.to_datetime(data['Start Date'])
data['End Date'] = pd.to_datetime(data['End Date'])

# Date filter
start_date = st.sidebar.date_input("Start Date", value=data['Start Date'].min())
end_date = st.sidebar.date_input("End Date", value=data['Start Date'].max())

filtered_data = data[
    (data['Start Date'] >= pd.to_datetime(start_date)) & (data['End Date'] <= pd.to_datetime(end_date))]

# Calculate MRR
filtered_data.loc[:, 'Month'] = filtered_data['Start Date'].dt.to_period('M')
mrr_data = filtered_data.groupby('Month').agg({'Amount': 'sum'}).reset_index()
mrr_data['Month'] = mrr_data['Month'].dt.to_timestamp()

# Calculate churn rate
churn_data = filtered_data[filtered_data['Status'] == 'canceled']
churn_data.loc[:, 'Churn Month'] = churn_data['End Date'].dt.to_period('M')
churn_rate = churn_data.groupby('Churn Month').size().reset_index(name='Churn Count')
churn_rate['Churn Month'] = churn_rate['Churn Month'].dt.to_timestamp()

# Calculate CLTV
filtered_data.loc[:, 'Lifetime Value'] = filtered_data['Amount'] * 12  # Assuming a simple yearly value
cltv = filtered_data['Lifetime Value'].mean()

# Additional Metrics
active_customers = filtered_data[filtered_data['Status'] == 'active']['Customer ID'].nunique()
total_revenue = filtered_data['Amount'].sum()
average_revenue_per_user = filtered_data['Amount'].mean()
churn_rate_percent = churn_data.shape[0] / filtered_data.shape[0] * 100
average_subscription_length = (filtered_data['End Date'] - filtered_data['Start Date']).mean().days

# Display Metrics
st.header("Key Metrics")
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly Recurring Revenue (MRR)", f"${mrr_data['Amount'].sum():,.2f}")
    with col2:
        st.metric("Customer Lifetime Value (CLTV)", f"${cltv:,.2f}")
    with col3:
        st.metric("Active Customers", active_customers)

    col4, col5 = st.columns(2)
    with col4:
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    with col5:
        st.metric("Average Revenue Per User (ARPU)", f"${average_revenue_per_user:,.2f}")

    col6, col7 = st.columns(2)
    with col6:
        st.metric("Churn Rate", f"{churn_rate_percent:.2f}%")
    with col7:
        st.metric("Average Subscription Length", f"{average_subscription_length:.2f} days")

# Graphs Section
st.header("Graphs")

# Plot MRR
fig_mrr = px.line(mrr_data, x='Month', y='Amount', title='Monthly Recurring Revenue (MRR)')
st.plotly_chart(fig_mrr)

# Plot churn rate
fig_churn = px.line(churn_rate, x='Churn Month', y='Churn Count', title='Monthly Churn Rate')
st.plotly_chart(fig_churn)

# Customer Segmentation by Plan
plan_data = filtered_data.groupby('Plan').agg({'Customer ID': 'count', 'Amount': 'sum'}).reset_index()
fig_plan = px.bar(plan_data, x='Plan', y='Amount', title='Revenue by Plan Type')
st.plotly_chart(fig_plan)

# Customer Growth Over Time
customer_growth = filtered_data.groupby('Month').size().reset_index(name='Customer Count')
customer_growth['Month'] = customer_growth['Month'].dt.to_timestamp()
fig_growth = px.line(customer_growth, x='Month', y='Customer Count', title='Customer Growth Over Time')
st.plotly_chart(fig_growth)

# Revenue Forecasting (Simple Linear Projection)
def revenue_forecast(data, months=12):
    data['Month_num'] = data['Month'].apply(lambda date: date.month + date.year * 12)
    X = data[['Month_num']]
    y = data['Amount']
    model = LinearRegression()
    model.fit(X, y)
    future_months = np.array(range(data['Month_num'].max() + 1, data['Month_num'].max() + 1 + months)).reshape(-1, 1)
    future_revenue = model.predict(future_months)
    future_dates = pd.date_range(start=data['Month'].max(), periods=months, freq='M')
    return pd.DataFrame({'Month': future_dates, 'Forecasted Revenue': future_revenue})

forecast_data = revenue_forecast(mrr_data, months=12)
fig_forecast = px.line(forecast_data, x='Month', y='Forecasted Revenue', title='Revenue Forecast')
st.plotly_chart(fig_forecast)

# Display Revenue by Region
region_data = filtered_data.groupby('Region').agg({'Customer ID': 'count', 'Amount': 'sum'}).reset_index()
fig_region = px.bar(region_data, x='Region', y='Amount', title='Revenue by Region')
st.plotly_chart(fig_region)

# 5-Point Analysis
st.header("5-Point Analysis")
analysis_points = [
    f"1. The total revenue generated during the selected period is ${total_revenue:,.2f}.",
    f"2. The average revenue per user (ARPU) is ${average_revenue_per_user:,.2f}.",
    f"3. The churn rate for the selected period is {churn_rate_percent:.2f}%.",
    f"4. The average subscription length is {average_subscription_length:.2f} days.",
    f"5. The forecasted revenue for the next 12 months shows an upward trend."
]
for point in analysis_points:
    st.write(point)

# PDF Report Generation
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'SaaS Metrics Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)

def create_pdf(mrr_data, churn_rate, customer_growth, forecast_data, analysis_points, figs):
    pdf = PDF()
    pdf.add_page()

    pdf.chapter_title("Monthly Recurring Revenue (MRR)")
    mrr_text = mrr_data.to_string(index=False)
    pdf.chapter_body(mrr_text)

    pdf.add_page()
    pdf.chapter_title("Churn Rate")
    churn_text = churn_rate.to_string(index=False)
    pdf.chapter_body(churn_text)

    pdf.add_page()
    pdf.chapter_title("Customer Growth Over Time")
    growth_text = customer_growth.to_string(index=False)
    pdf.chapter_body(growth_text)

    pdf.add_page()
    pdf.chapter_title("Revenue Forecast")
    forecast_text = forecast_data.to_string(index=False)
    pdf.chapter_body(forecast_text)

    pdf.add_page()
    pdf.chapter_title("5-Point Analysis")
    for point in analysis_points:
        pdf.chapter_body(point)

    # Add graphs to the PDF
    for fig in figs:
        pdf.add_page()
        pdf.image(fig, x=10, y=20, w=180)

    return pdf

# Save Plotly figures as images
figs = []
for fig in [fig_mrr, fig_churn, fig_plan, fig_growth, fig_forecast, fig_region]:
    fig_img = f"{fig.layout.title.text}.png"
    fig.write_image(fig_img)
    figs.append(fig_img)

# PDF Download Button
st.markdown('<div id="download"></div>', unsafe_allow_html=True)
if st.button("Download Report"):
    pdf = create_pdf(mrr_data, churn_rate, customer_growth, forecast_data, analysis_points, figs)
    pdf_file = "saas_metrics_report.pdf"
    pdf.output(pdf_file)
    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF", f, file_name=pdf_file)
