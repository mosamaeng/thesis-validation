import pandas as pd
import numpy as np
import psutil
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load train data from CSV file
train_df = pd.read_csv('dataset/clinical_mastitis_cows.csv')

train_df['Cow_ID'] = train_df['Cow_ID'].str.extract('(\d+)').astype(int)

# Create a dictionary to map breed values to numeric values
breed_map = {'jersey': 1, 'hostlene': 2}

# Use the map() method to replace breed values with numeric values
train_df['Breed'] = train_df['Breed'].map(breed_map)

# Split the train data into features and target
X_train = train_df.drop(columns=['class1'])
y_train = train_df['class1']

# Create an imputer to fill in missing values with the median of each column
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the train data
X_train = imputer.fit_transform(X_train)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train a KNN model on the train data
k = 3  # number of nearest neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Split the train data into chunks
chunk_size = 1000
X_chunks = np.array_split(X_train, len(X_train) // chunk_size)
y_chunks = np.array_split(y_train, len(y_train) // chunk_size)

# Train the model on each chunk
for i in range(len(X_chunks)):
    X_chunk = X_chunks[i]
    y_chunk = y_chunks[i]
    knn.fit(X_chunk, y_chunk)

# Load random data from CSV file
random_df = pd.read_csv('dataset/random_data.csv')

random_df['Cow_ID'] = random_df['Cow_ID'].str.extract('(\d+)').astype(int)

# Use the map() method to replace breed values with numeric values
random_df['Breed'] = random_df['Breed'].map(breed_map)

# Transform the random data using the same imputer and scaler
X_test = random_df
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)

# Make predictions on the random data
y_pred = knn.predict(X_test)

# Add the predicted class1 column to the random data
random_df['class1'] = y_pred

# Split the random data into features and target
X_test = random_df.drop(columns=['class1'])
y_test = random_df['class1']

# Transform the test data using the same imputer and scaler
X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)

# Make predictions on the test data and calculate accuracy score
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Get system memory usage
mem = psutil.virtual_memory()
print(f"Total Memory: {mem.total / (1024*1024*1024):.2f} GB")
print(f"Available Memory: {mem.available / (1024*1024*1024):.2f} GB")
print(f"Used Memory: {mem.used / (1024*1024*1024):.2f} GB")
print(f"Memory Percent Used: {mem.percent:.2f}%")

# Get process memory usage
process = psutil.Process()
mem_info = process.memory_info()
print(f"Process Memory: {mem_info.rss / (1024*1024):.2f} MB")

print(f"Accuracy on random data: {accuracy}")
