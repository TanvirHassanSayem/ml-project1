
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
plt.ion()

df = pd.read_csv('creditcard.csv')

def train_decision_tree_model(df):
    # Assuming 'Class' is the target column for fraud detection
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model



df.columns


df.head(3)

df.info()


df.describe()



df.isnull().sum()


sns.pairplot(df.sample(1000), hue='Class', palette='Set1')


from sklearn.model_selection import train_test_split

X = df.drop('Class',axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)



from sklearn.tree import DecisionTreeClassifier


dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)


dtree.fit(X_train,y_train)


predictions = dtree.predict(X_test)


predictions


from sklearn.metrics import classification_report,confusion_matrix


print(confusion_matrix(y_test,predictions))


from sklearn import tree
plt.figure(figsize=(20,25))
tree.plot_tree(dtree,feature_names=X.columns,class_names=['Class-1', 'Class-0'],rounded=True, # Rounded node edges
          filled=True, # Adds color according to class
          proportion=True
        )
plt.show()

