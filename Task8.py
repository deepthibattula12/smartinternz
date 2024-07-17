import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = sns.load_dataset('iris')
numeric_df = df.drop(columns=['species'])
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
linewidths=0.5)
plt.title('Correlation Matrix Heatmap of Iris Dataset', size=15)
plt.show()
