```python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:19:40 2024

@author: Student
"""

from sklearn import preprocessing as preP
from sklearn.preprocessing import MinMaxScaler



import numpy as np
x=np.array([2,4,6,8,10])
normalizedX=preP.normalize([x])
print("Normalized X in norm 2")
print(normalizedX)
normalizedX=preP.normalize([x],norm='max')
print("Normalized X in norm max")
print(normalizedX)
normalizedX=preP.normalize([x],norm='l1')
print("Normalized X in norm l1")
print(normalizedX)

x=np.array([10,12,15,20,22])
scaler=MinMaxScaler(feature_range=(10,15))
normalizedX=scaler.fit_transform(x.reshape(-1,1))

print(normalizedX)

x=np.array([10,12,15,20,22])
scaler=MinMaxScaler()
normalizedX=scaler.fit_transform(x.reshape(-1,1))

print(normalizedX)


x=np.array([[10],[12],[15],[20],[22]])
scaler=MinMaxScaler(feature_range=(10,15))
normalizedX=scaler.fit_transform(x)
```