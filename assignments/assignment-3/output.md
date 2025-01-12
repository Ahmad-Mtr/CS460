```
Original class distribution: Counter({1: 357, 0: 212})
Oversampled class distribution using RandomOversampling: Counter({0: 357, 1: 357})
Oversampled class distribution using SMOTE: Counter({0: 357, 1: 357})
Undersampled class distribution using RandomUndersampler: Counter({0: 212, 1: 212})
Undersampled class distribution using ClusterCentroids: Counter({0: 212, 1: 212})
Original class distribution after train_test_split:
Train Labels: Counter({1: 287, 0: 168})
Test Labels: Counter({1: 70, 0: 44})
Oversampled class distribution after train_test_split:
Train Labels: Counter({0: 289, 1: 282})
Test Labels: Counter({1: 75, 0: 68})
Undersampled class distribution after train_test_split:
Train Labels: Counter({1: 172, 0: 167})
Test Labels: Counter({0: 45, 1: 40})
==================================================
Accuracy Original DT: 0.9385964912280702
Precision Original DT: [0.93023256 0.94366197]
Recall Original DT: [0.90909091 0.95714286]
==================================================
Accuracy Oversampled DT: 0.8881118881118881
Precision Oversampled DT: [0.89393939 0.88311688]
Recall Oversampled DT: [0.86764706 0.90666667]
==================================================
Accuracy Undersampled DT: 0.9294117647058824
Precision Undersampled DT: [0.97560976 0.88636364]
Recall Undersampled DT: [0.88888889 0.975     ]
```