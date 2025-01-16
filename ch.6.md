# CH.6 Machine Learning Classification Cheat Sheet

## Types of Classification
### Binary Classification
- Predicts between two mutually exclusive categories
- Examples: `true/false`, `0/1`, `yes/no`
- Use case: Heart disease prediction (yes/no)

### Multi-Class Classification
- Predicts among multiple mutually exclusive categories
- Example: Image classification (car/plane/boat)

## Types of Learners
### Eager Learners
- Build model during training, faster predictions
- Examples:
	- Logistic Regression
	- Support Vector Machine
	- Decision Trees
	- Artificial Neural Networks

### Lazy Learners
- Memorize training data, no immediate model
- Slower predictions (compute at prediction time)
- Examples:
	- K-Nearest Neighbor

## Evaluation Metrics

### Confusion Matrix
- 2x2 matrix showing:
	- True Positives (TP)
	- True Negatives (TN)
	- False Positives (FP) - Type I Error
	- False Negatives (FN) - Type II Error

### Accuracy
$$
Accuracy = \frac{(TP + TN)}{Total observations}
$$
**When to use:**
- Balanced datasets
- Equal cost for FP and FN
- Equal benefit for TP and TN

### Precision
$$
Precision = \frac{TP}  {TP + FP}
$$

**When to use:**
- Cost of FP >> Cost of FN
- Benefit of TP >> Benefit of TN
- Question: "Out of all YES predictions, how many were correct?"

### Recall (Sensitivity)
$$
Recall = \frac{TP}  {TP + FN}
$$
**When to use:**
- Cost of FN >> Cost of FP
- Benefit of TN >> Benefit of TP
- Question: "How good at predicting real YES events?"

### Specificity
$$
Specificity =\frac{TN}{TN + FP}
$$
**When to use:**
- Question: "How good at predicting real NO events?"

### F1 Score
$$
F1\ Score = 2 * \frac{Precision * Recall} {Precision + Recall}
$$
**When to use:**
- Imbalanced datasets
- Need harmonic mean of precision and recall

### AUC-ROC Curve
**When to use:**
- Balanced datasets
- Need probability values instead of binary
- Shows trade-off between TP rate and FP rate

**Interpretation:**
- < 0.5: Poor classifier
- 0.5: Random classifier
- > 0.7: Good classifier
- > 0.8: Strong classifier
- 1.0: Perfect classifier

## Metric Selection Strategy
### Use ROC-AUC:
- For balanced datasets
- Need probability outputs

### Use Precision-Recall:
- For imbalanced datasets
- When false positives/negatives have different costs

### Use Accuracy:
- When costs/benefits of predictions are roughly equal
- For balanced datasets