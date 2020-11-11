
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import norm


# In[2]:


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


# In[3]:


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



# In[5]:


def DivisiveNormalization(theta, data):
    denom= theta[0] + np.linalg.norm((np.multiply(theta[1], data)),theta[2],1)
    v=np.divide(d.T, denom)
    
    return v


# In[12]:


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
    


# In[13]:


#This result has been spot checked against the values returned by the MATLAB code for data(1).X, cells 8, 10, and 14 
v=DivisiveNormalization(theta=theta, data=d)
pi=calcPiProbitQuad(Mi,v)


# In[14]:


print(pi)

