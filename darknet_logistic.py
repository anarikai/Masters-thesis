import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv('darknet2020.csv')

drop_columns = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']
data.drop(labels=drop_columns, axis=1, inplace=True)

label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])
data['Label.1'] = label_encoder.fit_transform(data['Label.1'])

data.fillna(value=0.0, inplace=True)
data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Label'], axis=1), data['Label'], test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('Darknet 2020, logisticRegression')
plt.show()

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print(report)


