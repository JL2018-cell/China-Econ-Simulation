import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# creating dataframe
num_industries = 6
upper_limit = [2000 for _ in range(num_industries)]
x = np.zeros(num_industries)
df = np.zeros((1, num_industries))
for i in range(num_industries):
    for j in range(upper_limit[i]):
        x[i] += 1
        df = np.concatenate((df, x.reshape(1, x.shape[0])), axis = 0)
print(df)
df = pd.DataFrame(df, columns = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism'])

# Importing standardscalar module
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

# fitting
scalar.fit(df)
scaled_data = scalar.transform(df)

# Importing PCA
from sklearn.decomposition import PCA

# Let's say, components = 2
pca = PCA(n_components = 2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

print("Transformed data shape:")
print(x_pca)

# giving a larger plot
plt.figure(figsize =(8, 6))

plt.scatter(x_pca[:, 0], x_pca[:, 1], c = np.arange(0, 1, 1/x_pca.shape[0]), cmap ='plasma')

# labeling x and y axes
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig("result_expt.png")

# components
print(pca.components_)

df_comp = pd.DataFrame(pca.components_, columns = df.columns)

plt.figure(figsize =(14, 6))

# plotting heatmap
sns.heatmap(df_comp)
plt.savefig("result_expt2.png")


