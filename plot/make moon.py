import matplotlib.pyplot as plt
from sklearn import manifold, datasets

from sklearn.cluster import KMeans

from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

plt.style.use('seaborn-dark')

# create the datase
#x, y = make_circles(n_samples=2000, factor=0.3, noise=0.06)
x, y = make_moons(n_samples=1000, noise=.05)

# normalization of the values
x = StandardScaler().fit_transform(x)

kmeans = KMeans(n_clusters=1).fit(x)
y2 = kmeans.predict(x)

# plot

# Plotting the generated samples
plt.plot(x[:, 0], x[:, 1], '.', c='lime',
         markeredgewidth=0.5,
         markeredgecolor='black')
plt.title(f'Spectre cluster')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.show()

