
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, \
    f1_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler

# Print the ratio for each class.
def calcRatio(dFrame):
    return dFrame.value_counts(normalize=True) *100



def getFeatures(data):
    X = data.iloc[:, 0:30]
    y = data['Class']

    trainData, testData, trainLabel, testLabel = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier()
    rf.fit(trainData, trainLabel)

    # Initialize Sequential Feature Selector
    sfs = SequentialFeatureSelector( rf, direction="forward", scoring="accuracy",  n_features_to_select=5)
    sfs.fit(trainData, trainLabel)
    selected_features = X.columns[sfs.get_support()].tolist()
    print("Selected Features:", selected_features)
    # ===========================
    # Step 3: Create a New Dataset with Selected Features
    # ===========================
    X_selected = data[selected_features]

    # Split the new dataset with test_size=0.33

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.33, random_state=42)

    # Initialize classifiers

    dt_classifier = DecisionTreeClassifier(random_state=42)

    knn_classifier = KNeighborsClassifier(n_neighbors=7)

    # Decision Tree Classifier

    print("\nDecision Tree Results:")

    print("-" * 50)

    dt_classifier.fit(X_train, y_train)

    dt_pred = dt_classifier.predict(X_test)

    dt_accuracy = accuracy_score(y_test, dt_pred)

    dt_precision = precision_score(y_test, dt_pred, average=None)

    dt_recall = recall_score(y_test, dt_pred, average=None)

    print(f"Accuracy: {dt_accuracy:.4f}")

    print("\nPrecision by class:")

    for i, p in enumerate(dt_precision):
        print(f"Class {i}: {p:.4f}")

    print("\nRecall by class:")

    for i, r in enumerate(dt_recall):
        print(f"Class {i}: {r:.4f}")



    # KNN Classifier

    print("\nKNN Results:")

    print("------------------------------------------")

    knn_classifier.fit(X_train, y_train)

    knn_pred = knn_classifier.predict(X_test)

    knn_accuracy = accuracy_score(y_test, knn_pred)

    knn_precision = precision_score(y_test, knn_pred, average=None)

    knn_recall = recall_score(y_test, knn_pred, average=None)


    print(f"Accuracy: {knn_accuracy:.4f}")

    print("\nPrecision by class:")

    for i, p in enumerate(knn_precision):
        print(f"Class {i}: {p:.4f}")

    print("\nRecall by class:")

    for i, r in enumerate(knn_recall):
        print(f"Class {i}: {r:.4f}")


    return selected_features




#df = pd.read_csv('creditcard.csv')


# Random
#randSample = df.sample(n=50000, random_state=42)
# randSample.to_csv('rand.csv', index=False)



randDF = pd.read_csv('rand.csv')

# sampleRatio= calcRatio(randDF['Class'])


# classCounts =calcRatio(df['Class'])
# print(classCounts)

# getFeatures(randDF, "Class", test_size=0.33, random_state=42, n_features_to_select=10)


# new_df = df[['Time', 'V1', 'V2', 'V3', 'V4']]
# new_df.to_csv('rand_selected.csv', index=False)
