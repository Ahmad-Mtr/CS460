import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier

def applyModel(name, x_train, x_test, y_train, y_test):
	# Model
	dt = DecisionTreeClassifier(random_state=42)
	dt.fit(x_train, y_train)
	dtPred = dt.predict(x_test)
	dtAccuracy = accuracy_score(y_test, dtPred)
	dtPrecision = precision_score(y_test, dtPred, average=None)
	dtRecall = recall_score(y_test, dtPred, average=None)
	
	dtF1 = f1_score(y_test, dtPred )
	print("="*50)
	print(f"Accuracy {name} DT: {dtAccuracy}")
	print(f"Precision {name} DT: {dtPrecision}")
	print(f"Recall {name} DT: {dtRecall}")

data = load_breast_cancer()
X=data.data
y= data.target

print("Original class distribution:", Counter(y))


""" 2. Under/Over-Sampling """
# Oversampling using RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
print("Oversampled class distribution using RandomOversampling:", Counter(y_over))

# Oversampling using SMOTE
s_oversample = SMOTE()
smote_X_over, smote_y_over = s_oversample.fit_resample(X, y)
print("Oversampled class distribution using SMOTE:", Counter(smote_y_over))

# Undersampling using RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(X, y)
print("Undersampled class distribution using RandomUndersampler:", Counter(y_under))

# Undersampling using ClusterCentroids
cc_undersample = ClusterCentroids(random_state=42)
cc_X_under, cc_y_under = cc_undersample.fit_resample(X, y)
print("Undersampled class distribution using ClusterCentroids:", Counter(cc_y_under))

""" 3. Train Test Split"""
# Original data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Oversampled data using SMOTE
over_x_train, over_x_test, over_y_train, over_y_test = train_test_split(smote_X_over, smote_y_over, test_size=0.2)

# Oversampled data using Random
rand_over_x_train, rand_over_x_test, rand_over_y_train, rand_over_y_test = train_test_split(X_over, y_over, test_size=0.2)

# Undersampled  data Random using ClusterCentroids
under_x_train, under_x_test, under_y_train, under_y_test = train_test_split(cc_X_under, cc_y_under, test_size=0.2)

# Undersampled  data Random using Random
rand_under_x_train, rand_under_x_test, rand_under_y_train, rand_under_y_test = train_test_split(X_under, y_under, test_size=0.2)



# Ratio of classes for both the training set and the test set in each case.
print("Original class distribution after train_test_split:")
print("Train Labels:", Counter(y_train))
print("Test Labels:", Counter(y_test))

print("Random Oversampled class distribution after train_test_split:")
print("Train Labels:", Counter(rand_over_y_train))
print("Test Labels:", Counter(rand_over_y_test))

print("SMOTE Oversampled class distribution after train_test_split:")
print("Train Labels:", Counter(over_y_train))
print("Test Labels:", Counter(over_y_test))

print("Random Undersampled class distribution after train_test_split:")
print("Train Labels:", Counter(rand_under_y_train))
print("Test Labels:", Counter(rand_under_y_test))

print("ClusterCentroid Undersampled class distribution after train_test_split:")
print("Train Labels:", Counter(under_y_train))
print("Test Labels:", Counter(under_y_test))


applyModel("Original", x_train, x_test, y_train, y_test)
applyModel("Oversampled using Random", rand_over_x_train, rand_over_x_test, rand_over_y_train, rand_over_y_test)
applyModel("Oversampled using SMOTE", over_x_train, over_x_test, over_y_train, over_y_test)
applyModel("Undersampled using Random", rand_under_x_train, rand_under_x_test, rand_under_y_train, rand_under_y_test )
applyModel("Undersampled using ClusterCentroid", under_x_train, under_x_test, under_y_train, under_y_test )
