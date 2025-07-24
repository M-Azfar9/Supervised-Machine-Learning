import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Load data
data = sns.load_dataset('titanic')

# Features and target
x = data.drop(['survived', 'fare'], axis=1)
y = data['survived']

# Handle missing values
x['age'] = x['age'].fillna(x['age'].mean())
x['deck'] = x['deck'].fillna(x['deck'].mode()[0])
x['embark_town'] = x['embark_town'].fillna(x['embark_town'].mode()[0])

# Split first
x_train_raw, x_test_raw, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2)

# Show class distribution
print("Train label distribution:\n", y_train.value_counts())
print("Test label distribution:\n", y_test.value_counts())

# One-hot encoding
categorical_cols = ['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Transform
x_train = preprocessor.fit_transform(x_train_raw)
x_test = preprocessor.transform(x_test_raw)

# Shape check
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

# Train models
lr = LogisticRegression(max_iter=2000)
dtc = DecisionTreeClassifier()
lr.fit(x_train, y_train)
dtc.fit(x_train, y_train)

# Predict
y_pred1 = lr.predict(x_test)
y_pred2 = dtc.predict(x_test)

# Dummy baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(x_train, y_train)
y_dummy = dummy.predict(x_test)

# Results
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred1))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred2))
print("Dummy Classifier Accuracy:", accuracy_score(y_test, y_dummy))
