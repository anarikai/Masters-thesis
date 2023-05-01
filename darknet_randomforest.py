import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("darknet2020.csv")
print(dataset.info())

# drop function which is used in removing or deleting rows or columns from the CSV files
dataset.drop('Flow ID', inplace=True, axis=1)
dataset.drop('Src IP', inplace=True, axis=1)
dataset.drop('Src Port', inplace=True, axis=1)
dataset.drop('Dst IP', inplace=True, axis=1)
dataset.drop('Dst Port', inplace=True, axis=1)
dataset.drop('Timestamp', inplace=True, axis=1)


# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()

dataset['Label'] = label_encoder.fit_transform(dataset['Label'])
#mapping = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))
#print(mapping)
dataset['Label.1'] = label_encoder.fit_transform(dataset['Label.1'])

print(dataset['Label'].unique())

#print(dataset.describe().T) # T transposes the table

y = dataset['Label']
X = dataset.drop(['Label'], axis=1)

X = np.nan_to_num(X)

scaler = MinMaxScaler()
model = scaler.fit(X)
X = model.transform(X)

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=SEED)

# creating bool series True for NaN values 
#bool_series = pd.isnull(dataset)


# Create our imputer to replace missing values with the mean e.g.
#X_train = X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
#X_train = X_train.fillna(X_train.mean(), inplace=True)
#imp.fit(X_train)

# Impute our data, then train
#X_train_imp = imp.transform(X_train)

rfc = RandomForestClassifier(n_estimators=900, 
                             max_depth=10,
                             random_state=SEED)

#X = X_test.fillna(X_test.mean())

# Fit RandomForestClassifier
rfc.fit(X_train, y_train)
# Predict the test set labels
y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('Random forest confusion matrix (0 = Non-Tor, 1 = NonVPN , 2 = Tor, 3 = VPN)')
plt.show()

print(classification_report(y_test,y_pred))

#g = sns.pairplot(dataset, hue='RiskLevel')
#g.fig.suptitle("Scatterplot and histogram of pairs of variables color coded by risk level", 
               #fontsize = 14, # defining the size of the title
               #y=1.05); # y = definig title y position (height)
#plt.show()  # <--- This is what you are looking for