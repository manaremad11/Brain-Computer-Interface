import numpy as np
from matplotlib import pyplot as plt
phase=90
amplitude=1
frequancy=1
n=np.arange(0,360,1)
n_sin=n+phase
n_cos=n+phase
n_sin=n_sin*frequancy
n_cos=n_cos*frequancy
sin=np.sin(n_sin*(np.pi/180))
cos=np.cos(n_cos*(np.pi/180))
sin=sin*amplitude
cos=cos*amplitude
plt.plot(n/360,sin,n/360,cos)
plt.show()