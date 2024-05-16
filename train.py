import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset
load_df = pd.read_csv("DATASET/PCOS_data_selected_features.csv")

X = load_df.drop('PCOS (Y/N)', axis=1)  # Features
y = load_df['PCOS (Y/N)']  # Target variable

# Step 4: Train-Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'Model/scaler.pkl')

# Step 6: Training with GridSearchCV to find optimal parameters
pipeline = Pipeline([
    ('clf', GaussianNB())
])

parameters = {
    'clf__var_smoothing': np.logspace(0,-9, num=100)
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Performance Evaluation
print("Best Parameters:", grid_search.best_params_)
best_nb_classifier = grid_search.best_estimator_

# Save the trained model
joblib.dump(best_nb_classifier, 'Model/naive_bayes_model.pkl')

# Predictions
y_pred = best_nb_classifier.predict(X_test_scaled)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))



_test_data = pd.DataFrame(X_test, columns=X.columns)
_test_data['label'] = y_test

_test_data.to_csv('test.csv', index=False)

