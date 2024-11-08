import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import requests

# Load the dataset
data = pd.read_csv(r'C:\\Users\\singh\\OneDrive\\Desktop\\Air-Quality\\sensor_data.csv')

# Checking the data
print(data.head())

# Handling missing values if any
data = data.dropna()

# Renaming columns to match actual data
data.columns = ['Temperature', 'Humidity', 'Timestamp']

# Converting 'Timestamp' to datetime, skipping invalid entries
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Drop rows where 'Timestamp' could not be converted
data = data.dropna(subset=['Timestamp'])

# Setting the 'Timestamp' as the index
data.set_index('Timestamp', inplace=True)

# Plot the data for initial visualization
data.plot(subplots=True, layout=(3, 2), figsize=(10, 8))
plt.show()

# If PM2.5 data is available, apply ARIMA
# ARIMA for PM2.5
# Check if the 'PM2.5' column exists
if 'PM2.5' in data.columns:
    pm25_data = data['PM2.5']

    # Split data into train and test sets
    train_size = int(len(pm25_data) * 0.8)
    train, test = pm25_data[:train_size], pm25_data[train_size:]

    # Fit the ARIMA model (using p, d, q parameters)
    model = ARIMA(train, order=(5, 1, 1))
    fitted_model = model.fit(disp=0)

    # Forecast
    forecast, stderr, conf_int = fitted_model.forecast(steps=len(test), alpha=0.05)

    # Plot actual vs forecasted values
    plt.plot(test.index, test, label='Actual PM2.5')
    plt.plot(test.index, forecast, label='Forecasted PM2.5', color='red')
    plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.show()
else:
    print("PM2.5 data not available in the dataset.")

# Function to send data to ThingSpeak
def send_data_to_thingspeak(pm25, co, nh3, temp, humidity, api_key):
    url = 'https://api.thingspeak.com/update'
    params = {
        'api_key': api_key,
        'field1': pm25,
        'field2': co,
        'field3': nh3,
        'field4': temp,
        'field5': humidity
    }
    response = requests.get(url, params=params)
    print(f'Status Code: {response.status_code}, Response: {response.text}')

# Example to send data
api_key = 'api key'
send_data_to_thingspeak(12.5, 0.4, 1.2, 30, 65, api_key)
