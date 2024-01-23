
import matplotlib.pyplot as plt
import numpy as np

time = np.arange(0, 125, 1/100)

#signals creating
signal_1 = np.random.uniform(-0.5,0.5,12500)
signal_2 = np.random.uniform(-0.5,0.5,12500)
plt.plot(time[0:100],signal_1[0:100],time[0:100],signal_2[0:100])
plt.show()

# trials reshaping
s1_trials=np.array(signal_1).reshape(50,-1)
s2_trials=np.array(signal_2).reshape(50,-1)

plt.plot(time[0:250],s1_trials[0])
plt.show()

#print(s1_trials.shape)
#print(s2_trials.shape)

#stimlous parting
s1_stimlous=s1_trials[:,0:50]
s2_stimlous=s2_trials[:,0:50]

#print(s1_stimlous.shape)
#print(s2_stimlous.shape)

#average for every stimlous
s1_stimlous_averages=np.sum(s1_stimlous,axis=1)/s1_stimlous.shape[1]
s2_stimlous_averages=np.sum(s2_stimlous,axis=1)/s2_stimlous.shape[1]

#print(s1_stimlous_averages.shape)
#print(s2_stimlous_averages.shape)

s1_stimloused=[]
s2_stimloused=[]


#creating numpy array of the same trials shape
#each row is one of trials stimlous mean repsented in 250 coulmn
for i in range(50):
    dum=[s1_stimlous_averages[i]]*250
    s1_stimloused.append(dum)
    dum=[s2_stimlous_averages[i]]*250
    s2_stimloused.append(dum)

s1_stimloused=np.array(s1_stimloused)
s2_stimloused=np.array(s2_stimloused)

#subtracting each stimous from its trial
s1_sup=np.subtract(s1_trials,s1_stimloused)
s2_sup=np.subtract(s2_trials,s2_stimloused)

plt.plot(time[0:250],s1_sup[0])
plt.show()

#average of all trials
s1_final=np.sum(s1_sup,axis=0)/s1_sup.shape[0]
s2_final=np.sum(s2_sup,axis=0)/s2_sup.shape[0]

print(s1_final.shape)
print(s2_final.shape)

plt.plot(time[0:250],s1_final[0:250],time[0:250],s2_final[0:250])
plt.show()

