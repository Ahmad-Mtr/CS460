```python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:04:03 2024

@author: Student
"""

import pandas as pd
import seaborn as sns

df=pd.read_csv("credit_scoring_Preprocessing.csv")
print(df.head())
print(df.info())
print(df.isna().sum()/len(df))
#sns.boxplot(df['children'])
print(df['children'].describe())
df['children']=abs(df['children'])
print(df['children'].describe())
df['children'].values[df['children']==20]=2
#sns.boxplot(df['days_employed'])
df['days_employed']=abs(df['days_employed'])
#sns.boxplot(df['days_employed'])
#Find median with outlier
'''
med=df['days_employed'].median()
df['days_employed']=df['days_employed'].fillna(med)
print(df.info())
df['days_employed'].values[df['days_employed']>=200000]=med
sns.boxplot(df['days_employed'])
print(df['days_employed'].describe())
'''
#Find median without outlier
y=df['days_employed']
indexT=df[df['days_employed']>=200000].index
y.drop(indexT,inplace=True)
print(df['days_employed'].iloc[13])
med=y.median()
df['days_employed'].values[df['days_employed']>=200000]=med
df['days_employed']=df['days_employed'].fillna(med)

print(df['days_employed'].iloc[13])
print(df['dob_years'].describe())

#check duplicate values
numOfduplicate=df.duplicated().sum()
df=df.drop_duplicates().reset_index(drop=True)
print(df.info())

df['days_employed']=round(df['days_employed'],0)
df.to_excel("newFile27_10_2024.xlsx",index=False)
df.to_csv("newFile27_10_2024.csv")
```