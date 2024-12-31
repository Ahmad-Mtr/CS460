import pandas as pd
import numpy as np
df = pd.read_csv('creditcard.csv')


# Random
randSample = df.sample(n=50000, random_state=42)
randSample.to_csv('rand.csv', index=False)

n = 50000/len(df)


# Strat
strat_sample=df.groupby('Class').apply(lambda x: x.sample(frac=n))
strat_sample.to_csv('strat.csv', index=False)


# Cluster
clusters = np.array_split(df, 4)

cClusters = np.random.choice(len(clusters), 2 , replace=False)
cSample = np.concatenate([clusters[i] for i in cClusters])
cdf = pd.DataFrame(cSample, columns=df.columns)

cdf.to_csv('cluster.csv', index=False)

