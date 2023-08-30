import numpy as np
import matplotlib.pyplot as plt
import math

features = 30
sigma_square = features
sigma = math.sqrt(sigma_square)

mu, sigma = 0, np.sqrt(sigma_square) # mean and standard deviation
scores = np.random.normal(mu, sigma, 10000)
scores = np.exp(scores)
sum = np.sum(scores)

x = np.linspace(-2*sigma, 2.0*sigma, num=1000)
y = np.exp(x) / sum

plt.plot(x, y)

plt.title("x's variance = # of feature in x = " + str(features))
plt.xlabel('x')
plt.ylabel('softmax(x)')

plt.ylim([0, 0.0005])

#show plot to user
plt.show()

a =5