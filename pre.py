import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# # Step 1: Data Preprocessing
# # Read the dataset
# df = pd.read_csv("DATASET/PCOS_data.csv")

# unwanted_columns = ['Unnamed: 44']  # Columns not needed for analysis
# df.drop(unwanted_columns, axis=1, inplace=True)

# missing_values = df.isnull().sum()
# print("Missing Values:")
# print(missing_values)

# # Remove rows with missing values
# df.dropna(inplace=True)

# print("*******************************")
# missing_values = df.isnull().sum()
# print("Missing Values:")
# print(missing_values)

# # Removing unwanted columns (if any)
# unwanted_columns = ['Sl. No', 'Patient File No.']  # Columns not needed for analysis
# df.drop(unwanted_columns, axis=1, inplace=True)

# print(df.head())

# # Encoding categorical variables
# label_encoder = LabelEncoder()
# for column in df.columns:
#     if df[column].dtype == np.object:  # check if the column is of object type (categorical)
#         df[column] = label_encoder.fit_transform(df[column])


# # Step 2: Feature Selection using Chi-Square Algorithm
# X = df.drop('PCOS (Y/N)', axis=1)  # Features
# y = df['PCOS (Y/N)']  # Target variable


# def perform_opt_feature_selection():

# 	best_features = SelectKBest(score_func=chi2, k='all')
# 	fit = best_features.fit(X, y)
# 	dfscores = pd.DataFrame(fit.scores_)
# 	dfcolumns = pd.DataFrame(X.columns)
# 	feature_scores = pd.concat([dfcolumns, dfscores], axis=1)
# 	feature_scores.columns = ['Feature', 'Score']

# 	# Display all features along with their scores
# 	print("Feature Scores:")
# 	print(feature_scores)

# 	# Select top 20 features
# 	k = 20
# 	selected_features = feature_scores.nlargest(k, 'Score')['Feature'].tolist()

# 	# Filter the dataset to keep only the selected features
# 	X_selected = X[selected_features]

# 	# Create a new DataFrame with the selected features
# 	df_selected = pd.concat([X_selected, y], axis=1)

# 	# Save the new DataFrame as a CSV file
# 	##df_selected.to_csv("DATASET/PCOS_data_selected_features.csv", index=False)

# perform_opt_feature_selection()


#EDA
load_df=pd.read_csv("DATASET/PCOS_data_selected_features.csv")

print(load_df.head())
print(load_df.info())
# Summary statistics
print(load_df.describe())

print(load_df['PCOS (Y/N)'].value_counts())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(load_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='PCOS (Y/N)', data=load_df)
plt.title('Countplot of PCOS (Y/N)')
plt.xlabel('PCOS (Y/N)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Count of target variable
class_counts = load_df['PCOS (Y/N)'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of PCOS (Y/N)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# Histogram for continuous variables
load_df.hist(figsize=(14, 14))
plt.suptitle('Histogram of Features', fontsize=16)
plt.show()

# Pairplot for visualization of relationships between variables
sns.pairplot(load_df, hue='PCOS (Y/N)')
plt.title('Pairplot of Variables')
plt.show()

# Boxplot for continuous variables
plt.figure(figsize=(16, 10))
sns.boxplot(data=load_df.drop('PCOS (Y/N)', axis=1))
plt.title('Boxplot of Features')
plt.xticks(rotation=45)
plt.show()

# Barplot for categorical variables
categorical_columns = ['Skin darkening (Y/N)', 'hair growth(Y/N)', 'Weight gain(Y/N)', 'Fast food (Y/N)', 'Cycle(R/I)', 'Pimples(Y/N)']
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=load_df, hue='PCOS (Y/N)')
    plt.title(f'Countplot of {column} vs PCOS')
    plt.xticks(rotation=45)
    plt.show()



