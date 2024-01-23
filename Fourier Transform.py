import numpy as np
from matplotlib import pyplot as plt

sample = 360
normalized_time = np.arange(0, sample) / sample


################################################
# signal creator
def mixed_signal(frequancy, phase, amplitude):
    signal = np.zeros(sample)
    for i in range(len(frequancy)):
        s = np.arange(0, sample, 1)
        s = s + phase[i]
        s = s * frequancy[i]
        signal = signal + amplitude[i] * np.sin(s * (np.pi / 180))
    return signal


################################################
# fourier transform function
def fourier(signal):
    N = sample
    n = np.arange(N)
    f = n.reshape((N, 1))
    t = n / N
    e = np.exp(-2j * np.pi * f * t)
    X = np.dot(e, signal)
    return X


################################################
# inverse fourier function
def inverse_fourier(coff):
    N = sample
    n = np.arange(N)
    f = n.reshape(N, 1)
    t = n / N
    e = np.exp(2j * np.pi * f * t)
    signal = np.dot(coff, e)
    return signal


#############################################
# creating signal
signal = mixed_signal([15, 20, 3, 8], [20, -50, 30, 0], [3, -0.5, 0.1, 1.6])

#############################################
# signal ploting
plt.plot(normalized_time, signal)
plt.title("ORIGINAL SIGNAL ")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

#############################################
# fourier transform
fourier_transform = fourier(signal)
coff = abs(fourier_transform * 2) / sample

#############################################
# spectrograph
plt.bar(np.arange(0, sample // 2, 1), coff[0:sample // 2])
plt.title("SPECTROGRAPH")
plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude')
plt.show()

#############################################
# inverse fourier
in_fourier = inverse_fourier(fourier_transform)

#############################################
# inverse ploting
plt.plot(normalized_time, in_fourier / sample)
plt.title("INVERSE SIGNAL")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
