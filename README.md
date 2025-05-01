# Energysage - Predictive Maintenance Dashboard

This is a Streamlit-based dashboard for predictive maintenance. The dashboard allows users to upload sensor data, visualize trends, predict failures, and configure alerts.

## Features

- **Upload Data**: Upload sensor data in CSV format.
- **Predictions**: Use a Random Forest model to predict failure probabilities.
- **Visualizations**: View failure distributions, historical trends, and correlation heatmaps.
- **Settings**: Configure dashboard settings such as themes, alert thresholds, and user preferences.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2.Create a virtual environment:
python -m venv venv

3. Activate the virtual environment:
On Windows:
venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate

4. Install the required dependencies:
pip install -r requirements.txt

5. Run the Streamlit app:
streamlit run dashboard.py

Open the app in your browser at http://localhost:8501.

Navigate through the sidebar options:

Upload Data: Upload your sensor data in CSV format.
Predictions: View failure predictions and configure alerts.
Visualizations: Explore failure distributions, trends, and correlations.
Settings: Customize the dashboard settings.