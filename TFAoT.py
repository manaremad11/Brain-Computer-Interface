import random as ra

import matplotlib.pyplot as plt

import numpy as np

windows_c = 0


# signal creator
def random_signal(s_l):
    signal = []
    for i in range(s_l):
        signal.append(ra.uniform(-0.5, 0.5))
    return signal

def mixed_signal(frequancy,phase,amplitude,num_points):
    signal=np.zeros(num_points)
    for i in range(len(frequancy)):
        s=np.arange(0,num_points,1)
        s=s+phase[i]
        s=s*frequancy[i]
        signal=signal+amplitude[i]*np.sin(s*(np.pi/180))
        #signal=signal+amplitude[i]*np.sin(s)
    return signal


# Fourier transform
def fourier(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape(N, 1)
    t = n / N
    e = np.exp(-2j * np.pi * k * t)
    X = np.dot(e, signal)
    return X


# Base line correction
def base_line(signal, length):
    base = signal[0:length]
    signal = signal - np.sum(base)
    return signal


# Window divider
def sliding_window(signal, w_size, overlap):
    windows = []
    right = w_size
    left = 0
    global windows_c
    while right != len(signal):
        windows.append(signal[left:right])
        left += overlap
        right += overlap
        windows_c += 1
    return np.array(windows)


window = 10
ove = 5
sampling_rate=100

# signal
Signal_1 = random_signal(350)
Signal_2 = random_signal(350)
Signal_3 = random_signal(350)
#Signal = mixed_signal([10,2,5,3],[0,0,0,0],[1,2,0.5,6],350)

# window division
W_signal_1 = sliding_window(Signal_1, window, ove)
W_signal_2 = sliding_window(Signal_2, window, ove)
W_signal_3 = sliding_window(Signal_3, window, ove)
F_signal_1 = []
F_signal_2 = []
F_signal_3 = []

# applying fourier
for i in W_signal_1:
    F_signal_1.append((fourier(i) * 2) / window)

for i in W_signal_2:
    F_signal_2.append((fourier(i) * 2) / window)
for i in W_signal_3:
    F_signal_3.append((fourier(i) * 2) / window)

F_signal = np.array(F_signal_1)

base_length=680//5

F_signal=np.array(base_line(np.array(F_signal).reshape(680),base_length))
magn= abs(F_signal)
power=magn**2
power=((power-np.sum(power[0:base_length]))/np.sum(power[0:base_length]))*100
power=np.array(power).reshape(68,10)

plt.pcolormesh(magn.reshape(68,10))
plt.show()
plt.pcolormesh(power)
plt.show()



three=[]
three.append(F_signal_1)
three.append(F_signal_2)
three.append(F_signal_3)
three=np.array(three)

angel=np.angle(three)
angel=np.exp(1j * angel)
angel=np.sum(angel,axis=0)/angel.shape[0]
angel=np.abs(angel)

print(F_signal.shape)

plt.pcolormesh(angel)
plt.show()
