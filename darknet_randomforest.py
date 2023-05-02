import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Read dataset and print info
dataset = pd.read_csv("darknet2020.csv")
print(dataset.info())

# Drop function which is used in removing or deleting rows or columns from the CSV files
dataset.drop('Flow ID', inplace=True, axis=1)
dataset.drop('Src IP', inplace=True, axis=1)
dataset.drop('Src Port', inplace=True, axis=1)
dataset.drop('Dst IP', inplace=True, axis=1)
dataset.drop('Dst Port', inplace=True, axis=1)
dataset.drop('Timestamp', inplace=True, axis=1)


# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()

# Transform Label and Label.1 columns to numbers
dataset['Label'] = label_encoder.fit_transform(dataset['Label'])
dataset['Label.1'] = label_encoder.fit_transform(dataset['Label.1'])

# Print Label column's unique values with unique()
print(dataset['Label'].unique())

# Divide the data into training stuff for the model, features or X, and what we want to predict, labels or Y
y = dataset['Label']
X = dataset.drop(['Label'], axis=1)

# Replace NaN with 0 and inf with finite numbers
X = np.nan_to_num(X)

# Transform features by scaling each feature to a given range
scaler = MinMaxScaler()
model = scaler.fit(X)
X = model.transform(X)

# train_test_split() method divides X and Y sets further into train & test sets
# Set random_state to SEED to make results reproducible
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=SEED)

# Create the model and create a forest with 900 trees and with each having 11 levels by setting max_depth to 10
rfc = RandomForestClassifier(n_estimators=900, 
                             max_depth=10,
                             random_state=SEED)


# Fit RandomForestClassifier
rfc.fit(X_train, y_train)

# Predict the test set labels
y_pred = rfc.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('Random forest confusion matrix (0 = Non-Tor, 1 = NonVPN , 2 = Tor, 3 = VPN)')
plt.show()

print(classification_report(y_test,y_pred))
