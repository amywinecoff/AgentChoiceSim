#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
from scipy.stats import norm


# In[83]:


#create a vector of the WTP values. These values are from data(1).X, cells 8, 10, and 14
d = np.array([
            [4, 2.33, 1.875, 1.8, 1.5, 1.495, 1.335, 1.275, 1.125, 1.09, 1, 0.925],
            [2.125, 2.125, 2.025, 2.0, 1.875, 1.495, 1.485, 1.335, 1.275, 1.075, 1.0, 0.625],
            [4.0, 2.17,  2.0, 2.0, 1.875, 1.875, 1.5, 1.485, 1.335, 1.275, 1.09, 1.075],
            ])

#These are the chosen options for s=1, on trials 8, 10,14
y=np.array([4,3,2])

#Choice set size for trials 8,10,14 for subject 1
Jm=12

#sigma, omega(w), beta
theta = [0.0000, 0.2376, 0.9739]


# In[84]:


#preallocate image matrices for choices
#This pertains to estimating covariance matrices of the error differences
#See Train book on discrete choice analysis p 113
#"This matrix can be used to transform the covariance matrix of
#errors into the covariance matrix of error differences: ~Ωi = MiΩMi.T .
temp = np.identity(Jm-1)
M = np.empty((Jm, Jm-1, 12))
for i in range(1, Jm+1):
    M[i-1] = np.concatenate((temp[:,0:i-1], -1*np.ones((Jm-1,1)), temp[:, i-1:]), axis=1)

#Matrices for only the chosen options
Mi=M[y-1]


# In[85]:


def DivisiveNormalization(theta, data):
    denom = theta[0] + np.multiply(theta[1], np.linalg.norm(data, theta[2], 1))
    v=np.divide(data.T, denom)
    
    return v


# In[86]:


def calcPiProbitQuad(Mi, v):
    
    MiT=np.transpose(Mi, axes=(0,2,1))
    T=v.shape[0]
    [x, w] = np.polynomial.hermite.hermgauss(100)

    #I honestly don't really know how tensordot works, but these lines of code return the correct values
    c = np.tensordot(MiT,v, axes=([1,0]))
    cT=np.transpose(c, axes=(0,2,1))
    vi = cT.diagonal() #This matches vi in MATLAB for s=1, trials 8,10,14
    
    #first part of equation in ProbaChoice.m, line 242
    z1=np.multiply(-2**0.5, vi)

    #second part of equation in ProbaChoice.m, line 242
    z2=np.multiply(-2**0.5, x)

    #These values have been validated
    zz = [z1-ele for ele in z2]

    aa=np.prod(norm.cdf(zz), axis=1)
    #Pi have been validated
    Pi=np.divide(np.sum(np.multiply(w.reshape(100,1), aa), axis=0), np.pi**0.5)
    
    return Pi
    


# In[87]:


#This result has been spot checked against the values returned by the MATLAB code for data(1).X, cells 8, 10, and 14 
v=DivisiveNormalization(theta=theta, data=d)
pi=calcPiProbitQuad(Mi,v)


# In[88]:


print(pi)


# # Simulation of user choices
# We simulate 100,000 trials of each of the 3 choice sets and use the values yielded by the `DivisiveNormalization` method + a random noise vector and check that the choice probabilities are roughly in line with the analytic probabilities from `calcPiProbitQuad`.

# In[ ]:


freq_chosen = np.array([0., 0., 0.])
num_it = 100000
v = DivisiveNormalization(theta=theta, data=d)
# the following covariance matrix has the structure
# [ 1    0.5    ...    0.5 ]
# [ 0.5    1    ...    0.5 ]
# [ 0.5   ...    1    0.5  ]
# [ 0.5   0.5   ...    1   ]
cov = np.ones((12, 12)) * 0.5
cov[np.arange(12), np.arange(12)] = 1
mean = np.zeros(12)
for i in range(num_it):
    eps = np.random.multivariate_normal(mean, cov, size=3).T
    u = v + eps
    item_chosen = (u.argmax(axis=0) == (y-1)).astype(float)
    freq_chosen += item_chosen / num_it
    
print(freq_chosen)


# In[ ]:


# save as Python
get_ipython().system('jupyter nbconvert --to script DivisiveNormalization.ipynb')

