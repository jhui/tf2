# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = 6, 6

# Initializing the random seed
random_seed = 1500

# Setting mean of the distributino to
# be at (0,0)
mean = np.array([0, 0])

# Iterating over different covariance
# values
cov_val = 0

# Initializing the covariance matrix
cov = np.array([[0.1, cov_val], [cov_val, 0.1]])

# Generating a Gaussian bivariate distribution
# with given mean and covariance matrix
distr = multivariate_normal(cov=cov, mean=mean,
                            seed=random_seed)

# Generating 5000 samples out of the
# distribution
data = distr.rvs(size=2000)

# Plotting the generated samples
plt.plot(data[:, 0], data[:, 1], '.', c='lime',
         markeredgewidth=0.5,
         markeredgecolor='black')
plt.title(f'Covariance between x1 and x2 = {cov_val}')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')

plt.show()