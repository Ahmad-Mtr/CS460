```python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:16:38 2024

@author: Rana.Almahmoud
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("data.csv")

plt.figure()
plt.hist(df['Gender'],color='red',bins=2,histtype='stepfilled')
plt.show



plt.figure()
plt.bar(df['Name'],df['Project_Grade'],color="#76FF7B")
plt.xlabel("students Name")
plt.ylabel("Project Grades")
plt.xticks(df['Name'],rotation=90)
plt.title("Grades for students")
plt.show()


plt.figure()
genderCount=df.groupby(['Gender','Internship_Status']).size().unstack()
genderCount.plot(kind='bar',stacked=True,color=['blue','purple'])

print(genderCount)


plt.figure()
plt.boxplot([df['Exam_Score'],df['Project_Grade']],labels=['Exam_Score','Project_Grade'])
plt.show()

plt.figure()
count=df['Internship_Status'].value_counts()
plt.pie(count,labels=count.index,autopct="%1.2f%%",colors=['red','green'])
plt.show()

mean=df['Exam_Score'].mean()
print("Mean =",mean)
max=df['Project_Grade'].max()
print("max =",max)
median=df['Exam_Score'].median

var=df['Exam_Score'].var()
```