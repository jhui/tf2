import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats


mean0 = [ 10, 10]
cov0  = [[4, 0], [0, 4]]
mean1 = [ 25, 30]
cov1  = [[ 2, 0], [0, 2]]


x = np.linspace(0, 40, 100)
y = np.linspace(0, 40, 100)
X, Y = np.meshgrid(x, y)
Z0 = np.random.random((len(x),len(y)))
Z1 = np.random.random((len(x),len(y)))

def pdf0(arg1,arg2):
    return (stats.multivariate_normal.pdf((arg1,arg2), mean0, cov0))
def pdf1(arg1,arg2):
    return (stats.multivariate_normal.pdf((arg1,arg2), mean1, cov1))
def pdf2(arg1,arg2):
    return (stats.multivariate_normal.pdf((arg1,arg2), mean2, cov2))


for i in range (0, len(x)):
    for j in range(0,len(y)):
        Z0[i,j] = pdf0(x[i],y[j])
        Z1[i,j] = pdf1(x[i],y[j])

Z0=Z0.T
Z1=Z1.T

# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111)
# ax3.contour(X,Y,Z0)
# ax3.contour(X,Y,Z1)
# ax3.contour(X,Y,Z2)
# plt.show()
#
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

surf1 = ax.plot_surface(X, Y, Z0+Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.contour(X, Y, Z0+Z1, zdir='z', offset=0)


ax.set_zlim(0, 0.1)

plt.show()