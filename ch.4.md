# Feature Engineering & Selection Cheatsheet

## Core Concepts

### Feature Definition
- Features = Input attributes/variables used in machine learning
- Examples:
  - Computer Vision: Lines in an image
  - NLP: Word count in a document
  - Tabular Data: Columns in dataset

## Feature Engineering

### 1. Main Processes

1. Feature Creation
   - Combines existing features
   - Requires human creativity
   - Methods: Addition, subtraction, ratios

2. Transformations
   - Adjusts predictor variables
   - Ensures consistent scale
   - Improves model understanding

3. Feature Extraction
   - Automated process
   - Reduces data volume
   - Methods:
     - Cluster analysis
     - Text analytics
     - Edge detection
     - PCA (Principal Component Analysis)

4. Feature Selection
   - Removes redundant/irrelevant features
   - Improves model performance


## Feature Selection Methods

### 1. Filter Methods
```python
# Example: Information Gain calculation
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris

# Load the Iris dataset 
iris = load_iris()
X, y = iris.data, iris.target

info_gain = mutual_info_classif(X, y) # Returns an array of scores, where each score corresponds to the **information gain** of a feature.

# Output
# Information Gain for each feature: [0.51929882 0.21639134 0.98069184 0.97782036]
```

#### Common Techniques:
1. **Information Gain**
   - Measures entropy reduction
   - Range: 0 to 1 (higher = better)

2. **Chi-square Test**
   - For categorical variables
   - Tests relationship with target

3. **Fisher's Score**
   - Supervised technique
   - Ranks variables by importance

4. **Missing Value Ratio**
   - Threshold-based evaluation
   - Formula: `missing_values / total_observations`

### 2. Wrapper Methods
```python
# Example: Sequential Feature Selection
from sklearn.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(
    estimator=model,
    direction="backward",
    n_features_to_select=2
)
```

#### Approaches:
1. **Forward Selection**
   - Starts empty, adds best features
   - Iterative process
   
2. **Backward Elimination**
   - Starts full, removes worst features
   - Removes one at a time

## Benefits of Feature Selection

1. **Performance**
   - Faster training
   - Improved accuracy
   - Reduced overfitting

2. **Simplicity**
   - Reduced model complexity
   - Better interpretability
   - Easier maintenance

## Best Practices

1. **Evaluation**
   - Test different selection methods
   - Validate results with cross-validation
   - Monitor model performance changes

2. **Implementation**
   - Start with filter methods (faster)
   - Use wrapper methods for refinement
   - Document selection criteria

3. **Maintenance**
   - Regular feature importance review
   - Update selection based on new data
   - Monitor feature correlations

Remember: The best feature selection method depends on your specific use case, data type, and computational resources.