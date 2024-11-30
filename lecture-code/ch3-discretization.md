```python
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:06:22 2024

@author: Student
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

x=[5,10,11,13,15,35,50,55,72,89,100,204,215 ]
print("Equal Width Discritization")

labels=pd.cut(x,bins=3,labels=['A','B','C'])
for v in labels :
 print(v," ")
    
print("Equal Frequency Discritiztion")

labels=pd.qcut(x,q=3,labels=['A','B','C'])
for v in labels :
    print(v," ")   
    
df=pd.read_excel("employees_jordan_data.xlsx")
print (df)


salary=df["Salary"]
labels=pd.cut(salary,bins=3,labels=['poor','Avg','Rich'])
df['SalaryLabel']=labels
df.to_excel("newSalariesLabels.xlsx")


salary=df["Salary"]
labels=pd.qcut(salary,q=3,labels=['poor','Avg','Rich'])
df['SalaryLabel']=labels
df.to_excel("newSalariesLabelsDependsOnFrequency.xlsx")


df=pd.read_excel("employees_jordan_data.xlsx")
dept=df["Department"]
le=LabelEncoder()
labeldDept=le.fit_transform(dept)
df["DepartmentLabel"]=labeldDept
df.to_excel("newLabeledDeptCol.xlsx")


df=pd.read_excel("employees_jordan_data.xlsx")
gender=df["Gender"]
le=LabelEncoder()
labelGender=le.fit_transform(gender)
df["Gender Label"]=labelGender
df.to_excel("newLabeledDeptCol.xlsx")
```