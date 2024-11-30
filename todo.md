### Ch.2
- [x] correlation
- [x] Mode
- [x] Plotly
- [x] Plotting function snippets

### Ch.3
- [x] `df.info()`
- [x] seaborn
- [x] `# check the percentage df.isna().sum() / len(df)`
Findings: The missing value percentage for both columns are around 10% 
```python
plt.figure(figsize=(10,6))
sns.heatmap(df.isna().transpose(), cmap="YlGnBu", cbar_kws={'label': 'Missing Data'})
```
- [ ] *Handling Outliers using sns*
- [x]  Handling Duplicates
- [x] Min-Max isn't the same as Slides
- [ ] Formulas for Min-Max & Z-Score
- [ ] 