import sys
import os.path
sys.path.append(	
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import dct
from scipy import linalg
import numpy as np
import time
import json
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["savefig.directory"] = 'C:\\Users\\Vandan\\Dropbox\\SURF Vandan'

with open('test/data_fast_proper_all.json') as data_file:    
    data = json.load(data_file)

# ns = np.arange(1,6)
# ns = ns**2
# ns *= 4
# ns = ns[:len(data['h2'])]
print(data['ns'])
#ns = data['ns'][0:-1]
ns = data['ns']
data.pop('ns')
print(ns)

keys = ['sls_fast','sls','h2','sls_slow']

for i in range(0,len(data)):
	key = keys[i]
	if(data[key] is not None):
		plt.plot(ns,data[key],label=key)




# plt.loglog(ns,data['sls_short'],label='sls columnwise')
# plt.loglog(ns,data['sls_slow_short'],label='sls')
# plt.loglog(ns,data['h2_short'],label='h2')
plt.xlabel("Size of chain network")
plt.ylabel("Controller computation time /s")
plt.title("The effect of column-wise decomposition on run time")
plt.legend()
plt.show()