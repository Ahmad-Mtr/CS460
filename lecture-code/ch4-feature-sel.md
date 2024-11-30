```python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:33:07 2024

@author: Rana.Almahmoud
"""

from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif,SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
df=load_iris()
data =df.data
label=df.target

newDF=pd.DataFrame(data,columns=df.feature_names)
newDF['Class']=label
newDF.to_excel("Iris.xlsx",index=False)

trainData,testData,trainLabel,testLabel=train_test_split(data,label\
                        ,test_size=0.3)

rf=RandomForestClassifier()
rf.fit(trainData,trainLabel)
y_prefiction=rf.predict(testData)
ac=accuracy_score(testLabel, y_prefiction)
print("Accuracy= ",ac)


#Filter Technique
selector=SelectKBest(score_func=mutual_info_classif,k=2)
x=selector.fit(data,label)
print(x)
selectedIndeices=x.get_support(indices=True)
print(selectedIndeices)
selectedFeature=newDF.columns[selectedIndeices]
print(selectedFeature)
selectedData=newDF.iloc[:,selectedIndeices]
print(selectedData)

trainData,testData,trainLabel,testLabel=train_test_split(selectedData,label\
                        ,test_size=0.3)

rf=RandomForestClassifier()
rf.fit(trainData,trainLabel)
y_prefiction=rf.predict(testData)
newAc=accuracy_score(testLabel, y_prefiction)

print("Old accuracy =",ac)
print("New accuracy =",newAc)

```