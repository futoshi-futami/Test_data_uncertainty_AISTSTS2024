# -*- coding: utf-8 -*-

import numpy as np


import matplotlib


matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

import seaborn as sns

sns.set()
def posterior(Phi, t, alpha, beta, return_inverse=False):
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)

    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N


def posterior_predictive(Phi_test, m_N, S_N, beta):
    y = Phi_test.dot(m_N)
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = 1 / beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    
    return y, y_var

def f(X, f_w0, f_w1,noise_variance):
    return f_w0 + f_w1 * X + noise(X.shape, noise_variance)


def g(X, noise_variance):
    return 0.5 + np.sin(2 * np.pi * X) + noise(X.shape, noise_variance)


def noise(size, variance):
    return np.random.normal(scale=np.sqrt(variance), size=size)


def identity_basis_function(x):
    return x


def gaussian_basis_function(x, mu, sigma=0.5):
    return np.exp(-0.5 * np.sum((x - mu) ** 2,-1) / sigma ** 2)


def polynomial_basis_function(x, power):
    return np.sum(x ** power,-1)


def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.concatenate((np.ones((1,x.shape[0],x.shape[1])),[bf(x, bf_arg) for bf_arg in bf_args]), axis=0)#np.concatenate([np.ones((x.shape[0],x.shape[1]))] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)
    
def funk(X, f_w0, f_w1,noise_variance):
    return f_w0 + f_w1 * X + noise(X.shape, noise_variance)


def p_U_znm(ALL_Phi_N,t):
    M=ALL_Phi_N.shape[1]
    S_u=0
    m_u=0
    for i in range(M):
        Phi=ALL_Phi_N[:,i,:]
        s_m_inv = beta**(-1)*np.identity(Phi.shape[1]) + alpha**(-1) * Phi.T.dot(Phi)
        
        s_m = np.linalg.inv(s_m_inv)
        m_u+=t[i].T @ s_m @ Phi.T
        S_u+=Phi @ s_m @ Phi.T
    S_u=np.identity(Phi.shape[0])+S_u
    m_u=np.linalg.inv(S_u) @ m_u.T
    
    return m_u,S_u
        
        

import matplotlib.pyplot as plt

dim=1
Ld=10

#Number of data points in each task
N=50
args=np.linspace(-2, 2, Ld)

#Number of meta-traning tasks
M_list = [1,2,5,10,20,50,100,200]#[i+1 for i in range(100)]

hyper_prior=np.random.normal(loc=-0.3,scale=1.0,size=[M_list[-1]+1,Ld+1])

task_parameters = np.random.normal(loc=hyper_prior,scale=1.0)



beta = 1.
alpha = 100.
# Test observations
X_test = np.random.normal(loc=0.0,scale=2.5,size=[1,100,dim])#np.random.rand(N_list[-1], 1) * 2 - 1#np.linspace(-1, 1, 100).reshape(-1, 1)
X_meta_train = np.random.normal(loc=0.0,scale=2.5,size=[1,N,dim])

# Design matrix of test observations
Phi_test = expand(X_test, bf=gaussian_basis_function, bf_args=args)
Phi_meta = expand(X_meta_train, bf=gaussian_basis_function, bf_args=args)

# Function values without noise 
y_true = np.sum(task_parameters[-1,:][:,None,None]*Phi_test,0)[:,:,None]+noise((np.sum(task_parameters[-1,:][:,None,None]*Phi_test,0)[:,:,None]).shape,1/beta)#f(X_test,task_parameters[-1,0], task_parameters[-1,0], noise_variance=1/beta)
y_meta_train = np.sum(task_parameters[-1,:][:,None,None]*Phi_meta,0)[:,:,None]+noise((np.sum(task_parameters[-1,:][:,None,None]*Phi_meta,0)[:,:,None]).shape,1/beta)#f(X_meta_train,task_parameters[-1,0], task_parameters[-1,0], noise_variance=1/beta)
    

    
# Training observations in [-1, 1)
ALLX = np.random.normal(loc=0.0,scale=2.5,size=[M_list[-1],N,dim])#np.random.rand(N_list[-1], 1) * 2 - 1
p0=expand(ALLX, bf=gaussian_basis_function, bf_args=args)
ALLt = (np.sum(task_parameters[:-1,:].T[:,:,None]*p0,0)[:,:,None])+noise(((np.sum(task_parameters[:-1,:].T[:,:,None]*p0,0)[:,:,None])).shape,1/beta)#f(ALLX,task_parameters[:-1,0][:,None,None], task_parameters[:-1,1][:,None,None], noise_variance=1/beta)

ALLX2 = np.random.normal(loc=0.0,scale=2.5,size=[M_list[-1],N,dim])#np.random.rand(N_list[-1], 1) * 2 - 1
p0=expand(ALLX2, bf=gaussian_basis_function, bf_args=args)
ALLt2 = (np.sum(task_parameters[:-1,:].T[:,:,None]*p0,0)[:,:,None])+noise(((np.sum(task_parameters[:-1,:].T[:,:,None]*p0,0)[:,:,None])).shape,1/beta)#f(ALLX2,task_parameters[:-1,0][:,None,None], task_parameters[:-1,1][:,None,None], noise_variance=1/beta)
# Training target values


# Training target values
Datam=[]
for n in M_list:
    X=ALLX[:n]
    t=ALLt[:n]
    X2=ALLX2[:n]
    t2=ALLt2[:n]

    # meta-hyper parameter learning
    ALL_Phi_N = expand(X, bf=gaussian_basis_function, bf_args=args)# expland (dim,M,N)
    m_u,S_u=p_U_znm(ALL_Phi_N,t)    
    S_u=np.linalg.inv(S_u)
    # meta parameter learning    

    
    ALL_m_N, ALL_S_N = posterior(Phi_meta.squeeze().T, y_meta_train.squeeze()[:,None], alpha, beta)
    y, y_var_ALL = posterior_predictive(Phi_test.squeeze().T, ALL_m_N, ALL_S_N, beta)
    
    P=Phi_meta.squeeze().T
    m_t=alpha*(m_u.T @ ALL_S_N @ Phi_test.squeeze()).T
    S_t=alpha*alpha*np.sum(((S_u @ ALL_S_N @ Phi_test.squeeze()))*(ALL_S_N @ Phi_test.squeeze()),0)#alpha * np.eye(P.shape[1]) + beta * P.T.dot(P)
    
    CMI=(np.log(y_var_ALL+S_t)-np.log(1/beta))/2
    
    MI_U=(-np.log(np.linalg.det(S_u))-np.log(1/alpha))/(2*n*N)
    
    MI_W=(-np.log(np.linalg.det(ALL_S_N)))/(2*N)
    
    
    A=0
    
    if n>=2:
    
        for m in range(n):
            X_N = np.concatenate((X[:m,:,:],X[m+1:,:,:]),axis=0)
            
            t_N = np.concatenate((t[:m,:,:],t[m+1:,:,:]),axis=0)
        
            partial_Phi_N = expand(X_N, bf=gaussian_basis_function, bf_args=args)# expland (dim,M,N)
            pm_u,pS_u=p_U_znm(partial_Phi_N,t_N)    
            pS_u=np.linalg.inv(pS_u)
            
            pS_t=alpha*alpha*np.sum(((pS_u @ ALL_S_N @ Phi_test.squeeze()))*(ALL_S_N @ Phi_test.squeeze()),0)
            CMI2=(np.log(y_var_ALL+pS_t)-np.log(1/beta))/2
            A+=np.mean(CMI2-CMI)



    Datam.append([np.mean(CMI),MI_U,MI_W,A/n])    
    





import matplotlib as mpl
mpl.style.use('default')
fig, ax = plt.subplots(1, 1,figsize=(4,3))
ax.plot(M_list,np.array(Datam)[:,0], linestyle = "solid",color="black",label=r"$I(Y;W|X,Z^N,Z^{NM})$")
ax.plot(M_list,np.array(Datam)[:,1], linestyle = "dashed",color="red",label=r"$I(U;Z^{NM})/NM$")
ax.plot(M_list,np.array(Datam)[:,2], linestyle = "dashed",color="purple",label=r"$I(W;Z^N|Z^{NM})/N$")
ax.plot(M_list[1:],np.array(Datam)[:,3][1:], linestyle = "dashed",color="blue",label=r"$I(Z^N;Z^{N,m}|Z^{N(M\backslash m)})$")

ax.set_xscale('log')
ax.set_yscale('log')
plt.tick_params(axis='y', which='minor')
ax.set_xlabel("M", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("meta_test2.eps")