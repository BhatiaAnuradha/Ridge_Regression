import numpy as np
import sys

import csv

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

#print np.dot(X_train.transpose(),y_train)
#print y_train
##print X_test
## Solution for Part 1
def part1():
    ## wRR = arg minw ||y-Xw||^2 + lambda||w||^2
    ## first need to define X_train and X_test matrices and y_train vector (inputs)
    ## then for loop for all the indices and finding the minimum of the argument which should be 0?
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    ## first need to compute XTX
    dummy = np.dot(X_train.transpose(),X_train)
    [n,m]= dummy.shape
    #print dummy.shape
    I = np.identity(n)
    dummy2 = lambda_input*I
    dummy3 = dummy2 + dummy
    dummy4 = np.dot(X_train.transpose(),y_train)
    wRR = np.dot(np.linalg.inv(dummy3),dummy4)
    return wRR
    #pass

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    ## need to compute the prior 
    ## cap_sigma = ((lambda_input*I)+ sigma-2(x0x0T+sum xixiT))^-1
    ## mu = ((lambda_input*sigma^2I)+(x0x0T+sum (xixiT)))^-1(x0y0+sum xiyi)

    #print cap_sig
    #print cap_sig.shape

    
    #train_mat_step = np.dot(X_train, X_train.transpose())
    #print train_mat_step
    #print train_mat_step.shape
    #for j in range (1,10):
    [p,q]= X_test.shape
        #active = np.zeros((1,10), dtype=np.int16)
        #print active.shape
        #print p,q
    #active = np.empty((1,10),int)
    active = []
    g = 1
    while (g < 11):
        for i in range (1,p):
            cov_mat_step = np.dot(X_train.transpose(), X_train)
            [u,v] = cov_mat_step.shape
            I2 = np.identity(u)
            cap_sig = np.linalg.inv((lambda_input*I2)+(cov_mat_step/sigma2_input))
            cov_mat_step_i = np.zeros([p,1])            #print cov_mat_step_i.shape
        
            #cov_mat_step_i[i,:] = 0
            X_test_i = X_test[i,:]
            #print X_test_i
            Ii = np.identity(i)
            #print Ii
            dummy_co = np.dot(X_test_i.T,cap_sig)
            cov_mat_step_i[i] = cov_mat_step_i[i]+(sigma2_input+(np.dot(dummy_co,X_test_i)))
            #print cov_mat_step_i
            
            sig_max = np.amax(cov_mat_step_i)
            #print np.amax(cov_mat_step_i)
            #print cov_mat_step_i.argmax(axis=0)
            if sig_max > sigma2_input:
                active = np.append(active, cov_mat_step_i.argmax(axis=0))
            g = g+1
            #print active.shape
            
        #if np.amax(cov_mat_step_i) > sigma2_input:
         #   active.append(cov_mat_step_i.argmax)
        
        
        
        #cap_sig_i = sum(X_train*X_train.transpose())
        #print cap_sig_i
    
    #print active
    #print active.shape
    return active
                              
                        
    #pass

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
