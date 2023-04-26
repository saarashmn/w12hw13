import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# generate data for a mixture model
np.random.seed(42)
k = 3
n = 500
mixing_coeffs = [0.4, 0.3, 0.3]
means = [[-1, 0], [1, 0], [0, 1]]
covs = [np.eye(2), np.eye(2), np.eye(2)]
labels = np.random.choice(k, size=n, p=mixing_coeffs)
X = np.vstack([np.random.multivariate_normal(means[i], covs[i], size=(labels == i).sum()) for i in range(k)])

kmeans = KMeans(n_clusters=k, random_state=42).fit(X)

np.random.seed(42)
Y = np.random.normal(loc=0, scale=1, size=n)


from scipy.stats import gaussian_kde

kde = gaussian_kde(Y)
y_grid = np.linspace(-5, 5, 500)
y_dens = kde(y_grid)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(y_grid, y_dens, color='blue', label='KDE estimate')
ax.hist(Y, density=True, bins=20, alpha=0.5, color='red', label='Dataset')
ax.set_xlabel('Y')
ax.set_ylabel('Density')
ax.legend()
plt.show()
