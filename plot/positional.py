import numpy as np
import matplotlib.pyplot as plt
import math



i = 32

base = 10000

pos = np.linspace(0, 50, num=1000)
y = pos
j = 2 * i / 512.0
omega = 1.0 / math.pow(base, j)

freq = omega / (2 * np.pi)
period  = round(1/freq, 2)
print(round(period, 2))
y = np.sin(pos * omega)
y2 = np.cos(pos * omega)
print(round(y[2], 2))
print(round(y2[2], 2))


plt.plot(pos, y2)

plt.title("Embedding: i=" + str(i) + ', period = ' + str(period))
plt.xlabel('pos')
plt.ylabel('positional')


#show plot to user
plt.show()

a =5