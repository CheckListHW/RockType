from matplotlib import pyplot as plt

import data_poro as data

plt.xscale('log')
plt.yscale('log')

plt.plot(data.poro_05, data.perm_05, ':', color='black')
plt.plot(data.poro_15, data.perm_15, ':', color='black')
plt.plot(data.poro_25, data.perm_25, ':', color='black')
plt.plot(data.poro_4, data.perm_4, ':', color='black')

plt.show()
