import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fpdf import FPDF
from sklearn.linear_model import LinearRegression

# Title of the dashboard
st.title("Simplified SaaS Metrics Dashboard")

# Generate Synthetic Data
def generate_synthetic_data():
    np.random.seed(42)
    n_customers = 500
    plan_types = ['Basic', 'Standard', 'Premium']
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa']

    customer_ids = np.arange(1, n_customers + 1)
    start_dates = pd.date_range(start="2022-01-01", periods=n_customers, freq='D')
    end_dates = [start + timedelta(days=np.random.randint(30, 365)) for start in start_dates]
    plans = np.random.choice(plan_types, n_customers, p=[0.4, 0.4, 0.2])
    regions = np.random.choice(regions, n_customers)

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
filtered_data['Month'] = filtered_data['Start Date'].dt.to_period('M')
mrr_data = filtered_data.groupby('Month').agg({'Amount': 'sum'}).reset_index()
mrr_data['Month'] = mrr_data['Month'].dt.to_timestamp()

# Calculate churn rate
churn_data = filtered_data[filtered_data['Status'] == 'canceled']
churn_data['Churn Month'] = churn_data['End Date'].dt.to_period('M')
churn_rate = churn_data.groupby('Churn Month').size().reset_index(name='Churn Count')
churn_rate['Churn Month'] = churn_rate['Churn Month'].dt.to_timestamp()

# Calculate CLTV
filtered_data['Lifetime Value'] = filtered_data['Amount'] * 12  # Assuming a simple yearly value
cltv = filtered_data['Lifetime Value'].mean()

# Additional Metrics
active_customers = filtered_data[filtered_data['Status'] == 'active']['Customer ID'].nunique()
total_revenue = filtered_data['Amount'].sum()
average_revenue_per_user = filtered_data['Amount'].mean()
churn_rate_percent = churn_data.shape[0] / filtered_data.shape[0] * 100
average_subscription_length = (filtered_data['End Date'] - filtered_data['Start Date']).mean().days

# Display Key Metrics
st.header("Key Metrics")
st.metric("Monthly Recurring Revenue (MRR)", f"${mrr_data['Amount'].sum():,.2f}")
st.metric("Customer Lifetime Value (CLTV)", f"${cltv:,.2f}")
st.metric("Active Customers", active_customers)
st.metric("Total Revenue", f"${total_revenue:,.2f}")
st.metric("Average Revenue Per User (ARPU)", f"${average_revenue_per_user:,.2f}")
st.metric("Churn Rate", f"{churn_rate_percent:.2f}%")
st.metric("Average Subscription Length", f"{average_subscription_length:.2f} days")

# Graphs Section
st.header("Graphs")

# Plot MRR
fig_mrr, ax_mrr = plt.subplots()
ax_mrr.plot(mrr_data['Month'], mrr_data['Amount'], marker='o')
ax_mrr.set_title('Monthly Recurring Revenue (MRR)')
ax_mrr.set_xlabel('Month')
ax_mrr.set_ylabel('Amount ($)')
st.pyplot(fig_mrr)

# Plot churn rate
fig_churn, ax_churn = plt.subplots()
ax_churn.plot(churn_rate['Churn Month'], churn_rate['Churn Count'], marker='o')
ax_churn.set_title('Monthly Churn Rate')
ax_churn.set_xlabel('Month')
ax_churn.set_ylabel('Churn Count')
st.pyplot(fig_churn)

# 5-Point Analysis
st.header("5-Point Analysis")
analysis_points = [
    f"1. Total revenue generated: ${total_revenue:,.2f}.",
    f"2. Average revenue per user (ARPU): ${average_revenue_per_user:,.2f}.",
    f"3. Churn rate: {churn_rate_percent:.2f}%.",
    f"4. Average subscription length: {average_subscription_length:.2f} days.",
    f"5. CLTV: ${cltv:,.2f}."
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

def create_pdf(mrr_data, churn_rate, analysis_points):
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
    pdf.chapter_title("5-Point Analysis")
    for point in analysis_points:
        pdf.chapter_body(point)

    return pdf

# PDF Download Button
st.markdown('<div id="download"></div>', unsafe_allow_html=True)
if st.button("Download Report"):
    pdf = create_pdf(mrr_data, churn_rate, analysis_points)
    pdf_file = "saas_metrics_report.pdf"
    pdf.output(pdf_file)
    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF", f.read(), file_name=pdf_file, mime='application/pdf')

