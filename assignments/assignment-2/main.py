
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SequentialFeatureSelector


# Print the ratio for each class.
def calcRatio(dFrame):
    return dFrame.value_counts(normalize=True) *100



def getFeatures(data):
    X = data.iloc[:, 0:30]
    y = data['Class']

    trainData, testData, trainLabel, testLabel = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier()
    rf.fit(trainData, trainLabel)

    # Initialize Feature Selector
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
    dtClassifier = DecisionTreeClassifier(random_state=42)
    knn_classifier = KNeighborsClassifier(n_neighbors=7)

    # Decision Tree Classifier
    print("\nDecision Tree Results:")
    print("------------------------------------+-")

    dtClassifier.fit(X_train, y_train)
    dtPred = dtClassifier.predict(X_test)

    dtAccuracy = accuracy_score(y_test, dtPred)
    dtPrecision = precision_score(y_test, dtPred, average=None)
    dtRecall = recall_score(y_test, dtPred, average=None)
    dtF1 = f1_score(y_test, dtPred )

    print(f"Accuracy: {dtAccuracy}")
    print(f"Precision DT: {dtPrecision}")
    print(f"Recall DT: {dtRecall}")
    print(f"F1 DT: {dtF1}")

    # KNN Classifier
    print("\nKNN Results:")
    print("------------------------------------------")

    knn_classifier.fit(X_train, y_train)
    knn_pred = knn_classifier.predict(X_test)

    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred, average=None)
    knn_recall = recall_score(y_test, knn_pred, average=None)
    knnF1 = f1_score(y_test, knn_pred )

    print(f"Accuracy KNN: {knn_accuracy}")
    print(f"Precision KNN: {knn_precision}")
    print(f"Recall KNN: {knn_recall}")
    print(f"F1 KNN: {knnF1}")



# df = pd.read_csv("creditcard.csv")
# print(calcRatio(df["Class"]))
# print("=====================================")


# Random
rdf = pd.read_csv("rand.csv")

print("Ratio rdf:")
print(calcRatio(rdf["Class"]))

getFeatures(rdf)

print("++++=========================================================+++++++")

# ٍStratified
sdf = pd.read_csv("strat.csv")

print("Ratio Stratified:")
print(calcRatio(sdf["Class"]))
getFeatures(sdf)

print("++++=========================================================+++++++")

# Cluster
cdf = pd.read_csv("cluster.csv")

print("Ratio Cluster:")
print(calcRatio(cdf["Class"]))

getFeatures(cdf)
