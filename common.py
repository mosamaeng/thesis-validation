import pandas as pd
import numpy as np

# Load data from CSV file
df = pd.read_csv('dataset/clinical_mastitis_cows.csv')

# Duplicate the data with random values
num_duplicates = 10000  # number of times to duplicate the data
num_cols = df.shape[1] - 1  # number of columns in the data (excluding last column)

# Create a new dataframe with random data
df_random = pd.DataFrame(columns=df.columns[:-1])

for col in df.columns[:-1]:
    if df[col].dtype == 'object':
        # If the column contains strings, generate random strings
        values = np.random.choice(df[col], size=num_duplicates)
        df_random[col] = values
    else:
        # If the column contains numeric data, generate random values
        values = np.random.randint(df[col].min(), df[col].max() + 1, size=num_duplicates)
        df_random[col] = values

# Save the new dataframe to a CSV file
df_random.to_csv('dataset/random_data.csv', index=False)
