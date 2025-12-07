import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# DataPrepare feature vectors
# --- 1. Load the Data ---
try:
    # Load data, ensuring timestamp and time columns are read as strings to start
    df = pd.read_csv('Contact Information (Responses) 8:9.csv')
except FileNotFoundError:
    print("Error: 'behavioral_data.csv' not found. Creating a dummy dataframe.")
    data = {
        'Timestamp': ['06/30/2025', '07/01/2025', '07/02/2025', '07/03/2025', '07/04/2025', '07/05/2025', '07/06/2025'],
        'Sleep': ['11:30:00 PM', '11:00:00 PM', '12:30:00 AM', '11:45:00 PM', '12:00:00 AM', '01:30:00 AM', '10:30:00 PM'],
        'Wake': ['07:00:00 AM', '06:30:00 AM', '08:00:00 AM', '07:15:00 AM', '08:30:00 AM', '09:30:00 AM', '07:00:00 AM'],
        'Waking Energy': [7, 6, 8, 7, 9, 8, 6],
        'Work': [8, 9.5, 7, 8.5, 2, 1, 9], # Lower work on Fri/Sat
        # Add other columns with placeholder values
        'NIC': [6, 0, 6, 0, 6, 0, 0], 'CAF': [200, 150, 250, 200, 50, 0, 100], 'ALC': [0, 1.5, 0, 0, 2.5, 3, 0]
    }
    # Add other columns from your heatmap here for the dummy data to run
    df_full = pd.DataFrame(data)
    for col in ['Sleep Quality', 'Focused Learning', 'Skill practicing', 'Physical Endeavors', 'Scrolling', 'Passive media', 'Active Media', 'Dominant Emotion intensity', 'System Architecture']:
        df_full[col] = np.random.rand(7) * 5
    df = df_full


# --- 2. Advanced Feature Engineering (Your Custom Code) ---
# Convert timestamp and time columns to datetime objects
df['Date'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y')
wake_time = pd.to_datetime(df['Wake'], format='%I:%M:%S %p')
sleep_time = pd.to_datetime(df['Sleep'], format='%I:%M:%S %p')

# Calculate core time-based features
df['hours_slept'] = ((wake_time - sleep_time).dt.seconds / 3600) % 24
df['bedtime_hour'] = sleep_time.dt.hour
df['wake_hour'] = wake_time.dt.hour

# Sort by date to enable temporal calculations
df = df.sort_values(by='Date').reset_index(drop=True)

# Create temporal and contextual features
df['Waking_Energy_Yesterday'] = df['Waking Energy'].shift(1)

# Create linear bedtime for accurate standard deviation calculation
cutoff_hour = 12
df['bedtime_hour_linear'] = df['bedtime_hour'].apply(lambda hour: hour + 24 if hour < cutoff_hour else hour)
df['bedtime_std_4_days'] = df['bedtime_hour_linear'].rolling(window=4).std()

# --- 3. Final Feature Selection & Cleaning ---
# Select only the final numerical columns for the model
# Exclude original timestamp/date/time strings and intermediate columns
feature_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in ['bedtime_hour_linear']]
df_features = df[feature_columns].copy()

# IMPORTANT: Rolling and shift operations create NaN values in the first few rows.
# The best practice is to drop these rows as they lack the complete temporal context.
df_features.dropna(inplace=True)

# --- 4. Normalization ---
scaler = MinMaxScaler()
S_total = scaler.fit_transform(df_features)

# --- 5. Final Output ---
print("Phase I complete with advanced feature engineering.")
print(f"Final features used for model: {df_features.columns.tolist()}")
print(f"Shape of the final feature matrix (S_total): {S_total.shape}")
print("\nSample of normalized feature vectors:")

print(S_total[0,:21])



# (At the very end of your data preparation)
print("Saving the S_total matrix to a CSV file...")
# We use np.savetxt because S_total is a NumPy array.
np.savetxt("s_total_features.csv", S_total, delimiter=",")
print("File saved successfully.")
np.savetxt("feature_names.csv", df_features.columns, fmt='%s')

print("Phase I complete.")

# --- The S_total variable is now available in memory ---

# (Code from Phase II: Manifold Learning)
# You can use S_total directly
