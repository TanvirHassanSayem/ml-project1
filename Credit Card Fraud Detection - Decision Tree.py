import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display initial information
print("Data Columns:", df.columns)
print("\nFirst 3 Rows:\n", df.head(3))
print("\nDataset Info:")
df.info()
print("\nDescriptive Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Visualize pairplot of a sample from the dataset
sns.pairplot(df.sample(1000), hue='Class', palette='Set1')
plt.show()

# Prepare data for training
X = df.drop(columns=['Class'])
y = df['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(X_train, y_train)

# Make predictions
predictions = dtree.predict(X_test)

# Display the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Plot the decision tree
plt.figure(figsize=(20, 15))
plot_tree(dtree, feature_names=X.columns, class_names=['Class-0', 'Class-1'],
          filled=True, rounded=True, proportion=True)
plt.title("Decision Tree Visualization")

# Use plt.pause() to ensure plot shows up
plt.draw()
plt.pause(5)  # Keeps the plot open for 5 seconds
plt.show()
