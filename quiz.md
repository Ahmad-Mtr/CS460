> [!NOTE] 
> The solution here scored 4.5/5, the `frac` values are wrong

```python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:46:59 2024

@author: Student
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_excel('data.xlsx')

# Random
rand_sample = data.sample(n=12, random_state=42)
rand_sample.to_excel('Random.xlsx', index=False)
plt.figure()
sampleRatio=rand_sample['Class'].value_counts(normalize=True)
plt.pie(sampleRatio,labels=sampleRatio.index,autopct="%1.2f%%")
plt.show()


# Stratified
strat_sample = data.groupby('Class').apply(lambda x: x.sample(frac=0.4))
strat_sample.to_excel('stratified.xlsx', index=False)

plt.figure()
sampleRatio1=strat_sample['Class'].value_counts(normalize=True)
plt.pie(sampleRatio1,labels=sampleRatio1.index,autopct="%1.2f%%")
plt.show()

# Clustered
cdf = data.sample(frac=1)

clusters = np.array_split(cdf, 3)

cClusters = np.random.choice(len(clusters), 1 , replace=False)
cSample = np.concatenate([clusters[i] for i in cClusters])
cdf = pd.DataFrame(cSample, columns=data.columns)
print(cdf)
cdf.to_excel('clustered.xlsx', index=False)

plt.figure()
sampleRatio2=cdf['Class'].value_counts(normalize=True)
plt.pie(sampleRatio2,labels=sampleRatio2.index,autopct="%1.2f%%")
plt.show()

# Diagram
'''
counts= data["Class"].value_counts()



plt.figure()
sample1= data.sample(12)
print(sample1)


sampleRatio=sample1['Class'].value_counts(normalize=True)
print(sampleRatio)

plt.pie(sampleRatio,labels=sampleRatio.index,autopct="%1.2f%%")
plt.show()



'''
```