import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Read dataset as data frame
df = pd.read_csv("darknet2020.csv")

print(df.info())
 
# Split dataset into testing and training sets
training_data = df.sample(frac=0.8, random_state=25)
testing_data = df.drop(training_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")

y_train = training_data['Label']
training_data.drop(labels='Label', axis=1, inplace=True)

full_data = training_data.append(testing_data)

drop_columns = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']
full_data.drop(labels=drop_columns, axis=1, inplace=True)

# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()

full_data['Label'] = label_encoder.fit_transform(full_data['Label'])
full_data['Label.1'] = label_encoder.fit_transform(full_data['Label.1'])

full_data.fillna(value=0.0, inplace=True)
full_data = full_data[~full_data.isin([np.nan, np.inf, -np.inf]).any(1)]

# split the data into training and testing sets
X_train = full_data.values[0:113224]
X_test = full_data.values[113224:]

# scale our data by creating an instance of the scaler and scaling it
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

state = 12  
test_size = 0.20  
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

#for learning_rate in lr_list:
    #gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    #gb_clf.fit(X_train, y_train)

    #print("Learning rate: ", learning_rate)
    #print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    #print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

cm = confusion_matrix(y_val, predictions)
sns.heatmap(cm, annot=True, fmt='d').set_title('Gradient boosting classifier')
plt.show()

print("Classification Report")
print(classification_report(y_val, predictions))