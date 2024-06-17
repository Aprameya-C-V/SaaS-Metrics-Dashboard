# SaaS Metrics Dashboard

## Overview
The SaaS Metrics Dashboard is a comprehensive tool for analyzing key metrics of a SaaS business. It includes visualization of Monthly Recurring Revenue (MRR), churn rate, customer growth, revenue by plan, and regional revenue distribution. The dashboard also provides a 5-point analysis and generates detailed PDF reports.

## Features
- **Key Metrics Display**: Monthly Recurring Revenue (MRR), Customer Lifetime Value (CLTV), Active Customers, Total Revenue, Average Revenue Per User (ARPU), Churn Rate, and Average Subscription Length.
- **Graphs**: Line charts for MRR, churn rate, customer growth, revenue forecast, and bar charts for revenue by plan and region.
- **Regional Map**: Geographical representation of revenue by region.
- **Custom Data Input**: Upload your own CSV data file to visualize metrics.
- **PDF Report Generation**: Downloadable PDF report with key metrics, graphs, and analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/saas-metrics-dashboard.git
   cd saas-metrics-dashboard
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run saas_dashboard.py
   ```

## Usage
1. Upload your CSV file via the sidebar. Ensure the file has the following columns:
   - Customer ID
   - Start Date
   - End Date
   - Amount
   - Status
   - Plan
   - Region

   Example:
   ```csv
   Customer ID,Start Date,End Date,Amount,Status,Plan,Region
   1,2022-01-01,2022-12-31,100,active,Standard,North America
   2,2022-02-15,2022-08-15,200,canceled,Premium,Europe
   ```

2. Use the sidebar to filter the data by date range.

3. View key metrics and graphs in the main dashboard.

4. Download the PDF report from the button at the bottom of the dashboard.
