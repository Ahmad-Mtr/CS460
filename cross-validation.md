Use this article for Cross Validation [Explanation](https://www.geeksforgeeks.org/cross-validation-machine-learning/)

---
Write a Python script that performs the following steps:

1. Load the `load_breast_cancer` dataset from `sklearn.datasets`.
2. Apply both **K-Nearest Neighbors (KNN)** and **Decision Tree** classifiers using 5-fold cross-validation.
3. Print the accuracy values for each fold and calculate the average accuracy for both classifiers.
---

### **Cheatsheet: Python Syntax for Cross Validation**

#### **1. Importing Required Libraries**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
```

#### **2. Loading the Dataset**

```python
data = load_breast_cancer()
X, y = data.data, data.target
```

#### **3. Setting Up Cross-Validation**

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
```

#### **4. Initializing Classifiers**

```python
knn = KNeighborsClassifier()
decision_tree = DecisionTreeClassifier(random_state=42)
```

#### **5. Applying Cross-Validation and Calculating Accuracy**

- **For K-Nearest Neighbors**:

```python
knn_accuracies = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
```

- **For Decision Tree**:

```python
dt_accuracies = cross_val_score(decision_tree, X, y, cv=kf, scoring='accuracy')
```

#### **6. Printing Accuracy for Each Fold**

```python
print("KNN Accuracies for each fold:", knn_accuracies)
print("Decision Tree Accuracies for each fold:", dt_accuracies)
```

#### **7. Calculating Average Accuracy**

```python
knn_avg_accuracy = np.mean(knn_accuracies)
dt_avg_accuracy = np.mean(dt_accuracies)

print("KNN Average Accuracy:", knn_avg_accuracy)
print("Decision Tree Average Accuracy:", dt_avg_accuracy)
```

---

### **Solution**

```python
from sklearn.datasets import load_breast_cancer  
from sklearn.model_selection import cross_val_score, KFold  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier  
import numpy as np  
  
data = load_breast_cancer()  
X, y = data.data, data.target  
  
kFold = KFold(n_splits=5, shuffle=True, random_state=42)  
  
knn = KNeighborsClassifier()  
decisionTree = DecisionTreeClassifier(random_state=42)  
  
knnAccuracies = cross_val_score(knn, X, y, cv=kFold, scoring='accuracy')  
dtAccuracies = cross_val_score(decisionTree, X, y, cv=kFold, scoring='accuracy')  
  
print("KNN Accuracies for each fold:", knnAccuracies)  
print("Decision Tree Accuracies for each fold:", dtAccuracies)  
  
avgKnn = np.mean(knnAccuracies)  
avgDt = np.mean(dtAccuracies)  
  
print(avgKnn, avgDt)  
  
if avgDt > avgKnn: print("Decision Tree is better")  
else: print("KNN is better")
```
