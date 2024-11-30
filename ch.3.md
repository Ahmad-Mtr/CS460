Here's a concise Data Mining cheatsheet focusing on Data Preprocessing in Python:

# Data Preprocessing in Python

## Core Concepts

- **Data Preprocessing**: Transforms raw data into analyzable format for machine learning
- **Importance**: Prevents "garbage in, garbage out" - bad data leads to poor model performance

## Essential Steps

### 1. Data Collection
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('your_file.csv')
```

---
### 2. Data Cleaning

#### Missing Values
```python
# Check missing values
df.isna().sum()

# Calculate missing value percentage
missing_percent = df.isna().sum() / len(df) * 100
print(missing_percent)

# Visualize Missing Data via a Heatmap
sns.heatmap(df.isna().transpose(), cmap="YlGnBu", 
	cbar_kws={'label': 'Missing Data'})
plt.show()


# Fill missing values (by median or others)
df['column'] = df['column'].fillna(df['column'].median())
```

#### Handling Outliers
```python
# Visualize outliers
sns.boxplot(df['column'])
plt.show()
```

###### Case 1: Replace Outlier Values with Median (including outliers in the calculation)

```python
# Replace outliers greater than 200,000 with the median
df['days_employed'].values[df['days_employed'].values > 200000] = df['days_employed'].median()
```

###### Case 2: Replace Outliers with Median (excluding outlier values in the calculation)

```python
# Identify indexes of outliers or missing values
indexAge = df[(df['days_employed'] > 200000) | (df['days_employed'].isnull())].index

# Drop these indexes temporarily to calculate the median
x = df['days_employed']
x.drop(indexAge, inplace=True)

# Replace outliers with the median of remaining values
df['days_employed'].values[(df['days_employed'].values > 200000) | (df['days_employed'].isnull())] = x.median()
```


#### Remove Duplicates
```python
df = df.drop_duplicates().reset_index(drop=True)
```
---
### 3. Data Normalization

#### Min-Max Normalization
 **Definition**: Min-Max Normalization is a data pre-processing technique that scales the values of a numerical feature to a specific range, usually `[0, 1]` or `[a, b]`. This method preserves the relationships among the original data values while transforming them into the desired range.

 **Formula**: For a given value $v$, the normalized value $v′$ is calculated as:

$$v' = \frac{v - \text{minA}}{\text{maxA} - \text{minA}} \times (\text{new\_maxA} - \text{new\_minA}) + \text{new\_minA}$$

- **minA**: Minimum value of the original attribute.
- **maxA**: Maximum value of the original attribute.
- **new_minA**: Minimum value of the new range.
- **new_maxA**: Maximum value of the new range.

###### **Example**

- **Original Attribute Range**: Income values between $12,000 and $98,000.
- **Desired Range**: `[0, 1]`.
- To normalize an income value of $73,000:

$$
v' = \frac{73000 - 12000}{98000 - 12000} \times (1 - 0) + 0 = 0.716
$$

###### **Code Example**

Here’s how to implement Min-Max Normalization in Python using `sklearn.preprocessing.MinMaxScaler`:

```python
from sklearn.preprocessing import MinMaxScaler

# Sample data
data = [[12000], [73000], [98000]]

# Initialize MinMaxScaler to scale data
scaler = MinMaxScaler()

# Fit and transform the data
normalized_data = scaler.fit_transform(data)

# Output the results
print("Original Data:", data)
print("Normalized Data:", normalized_data)
```

**Output**:

```
Original Data: [[12000], [73000], [98000]]
Normalized Data: [[0.  ], [0.70930233], [1.  ]]
```

###### **Advantages**

- Ensures all features contribute equally to the model.
- Useful for algorithms sensitive to data magnitude, such as distance-based methods (e.g., KNN).



#### Z-Score Normalization

 **Definition**: Z-Score Normalization (or zero-mean normalization) scales a numerical feature by centring it around the mean and scaling it by the standard deviation. It standardizes data to have a mean of 0 and a standard deviation of 1.

**Formula**: For a given value $X$, the z-score $Z$ is:

$$Z = \frac{X - \mu}{\sigma}$$

- **μ**: Mean of the feature.
- **σ**: Standard deviation of the feature.

###### **Example**

Given the dataset: `[6, 7, 7, 12, 13, 13, 15, 16, 19, 22]`,

- Mean (`μ`): 14.
- Standard Deviation (`σ`): 4.48.

The z-score for 22 is:

$$Z = \frac{22 - 14}{4.48} = 1.79$$

###### **Code Example**

```python
from scipy.stats import zscore
import numpy as np

# Sample data
data = np.array([6, 7, 7, 12, 13, 13, 15, 16, 19, 22])

# Calculate Z-scores
z_scores = zscore(data)

# Output results
print("Original Data:", data)
print("Z-Scores:", z_scores)
```

**Output**:

```
Original Data: [ 6  7  7 12 13 13 15 16 19 22]
Z-Scores: [-1.39, -1.195, -1.195, -0.199,  0., 0., 0.39, 0.59, 1.19, 1.79]
```

###### **Advantages**

- Removes the impact of scale and magnitude differences.
- Useful for algorithms like gradient descent or distance-based models.

---
### 4. Discretization

#### Equal-Width Binning
```python
# 3 Bins that have the same width
pd.cut(data, bins=3, labels=[1, 2, 3])
```

#### Equal-Frequency Binning
```python
# 3 Bins that have equal occurrences in each
pd.qcut(data, q=3, labels=[1, 2, 3])
```


---
### 5. Feature Engineering

#### Label Encoding
Convert categorical columns into numerical ones
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encoded_column = le.fit_transform(df['categorical_column'])
```
---

# Machine Learning Basics in Python

## Train-Test Split

### What is Train-Test Split?
- A fundamental technique to evaluate model performance on unseen data
- Splits dataset into two parts: training set and testing set
- Helps prevent overfitting and provides realistic model evaluation

### Why Use Train-Test Split?
1. **Model Validation**: Simulates how model performs on new, unseen data
2. **Prevents Overfitting**: By testing on separate data, we can detect if model is memorizing training data
3. **Performance Estimation**: Gives realistic estimate of model's real-world performance

### Implementation
```python
from sklearn.model_selection import train_test_split

# Split data (common split is 70-30 or 80-20)
X_train, X_test, y_train, y_test = train_test_split(
    features,      # Your input features
    target,        # Your target variable
    test_size=0.3, # 30% for testing
    random_state=0 # For reproducibility
)
```

### Best Practices
- Usually use 70-80% for training, 20-30% for testing
- Set random_state for reproducibility
- Ensure balanced class distribution in splits
- Consider stratification for imbalanced datasets

---
## Random Forest

### What is Random Forest?
- An ensemble learning method using multiple decision trees
- Combines predictions from many trees to make final prediction
- Built on the concept of "wisdom of crowds"

### Why Use Random Forest?
1. **Reduced Overfitting**: Combines multiple trees to reduce variance
2. **Feature Importance**: Provides built-in feature importance rankings
3. **Handles Non-linearity**: Can capture complex patterns in data
4. **Robust**: Works well with both numerical and categorical data

### Implementation
```python
from sklearn.ensemble import RandomForestClassifier

# Create and train the model
rf = RandomForestClassifier()

# Train the model
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

```

---
## Types of Features

1. **Categorical Features**
   - Values from predefined set
   - Examples: colors, months, True/False

2. **Numerical Features**
   - Continuous values
   - Examples: prices, counts, measurements

## Best Practices

- Check data quality before preprocessing
- Handle missing values appropriately for your use case
- Document preprocessing steps for reproducibility
- Validate results after each transformation
- Use consistent scaling methods across similar features