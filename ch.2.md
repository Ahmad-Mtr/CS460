### Exploratory Data Analysis Cheat Sheet: Chapter 2

#### Introduction to Python for Data Science

- **Why Python?**
    - Simple, versatile, and rich in libraries for data analysis.
    - Recommended: Use the **Anaconda distribution** for pre-installed libraries.

#### Essential Libraries

- **Pandas**: Data manipulation (e.g., `pd.read_csv()`, `pd.read_excel()`).
- **Matplotlib & Seaborn**: Data visualization.
---

#### Common Plots in Python

| **Plot Type**        | **Code Snippet**        | **Usage**                   |
| -------------------- | ----------------------- | --------------------------- |
| **Line Plot**        | `plt.plot()`            | Trends over time.           |
| **Scatter Plot**     | `plt.scatter()`         | Relationships between vars. |
| **Histogram**        | `plt.hist()`            | Frequency distribution.     |
| **Bar Plot**         | `plt.bar()`             | Compare categories.         |
| **Stacked Bar Plot** | `plt.bar(stacked=True)` | Compare sub-categories.     |
| **Box Plot**         | `plt.boxplot()`         | Distribution & outliers.    |
| **Pie Chart**        | `plt.pie()`             | Proportional comparison.    |

---
#### Descriptive Statistics

- **Mean**: Average value.
![Mean Formula](assets/mean-form.png)
- **Median**: Middle value (ordered dataset).
- **Mode**: Most frequent value(s).
- **Variance**: Dispersion of data around the mean. 
![Variance](assets/variance-form.png)
**Python Example**:

```python
exam_scores = data['Exam_Score']
print("Mean:", exam_scores.mean())
print("Median:", exam_scores.median())
print("Variance:", exam_scores.var())
```

---

#### Correlation Analysis

- Measures the relationship between variables.
    - **Positive Correlation**: Variables increase together.
    - **Negative Correlation**: One increases while the other decreases.
    - **No Correlation**: No clear relationship.

**Python Example**:

```python
correlation_matrix = data.corr()
print(correlation_matrix)
```

---

#### Visual Examples

1. **Line Plot**  
    ![Line Plot](https://via.placeholder.com/600x300?text=Line+Plot+Placeholder)
    
2. **Scatter Plot**  
    ![Scatter Plot](https://via.placeholder.com/600x300?text=Scatter+Plot+Placeholder)
    

---

