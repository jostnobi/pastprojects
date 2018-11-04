from math import *
import numpy as np
import random


#jb17969

#This code requires numpy and python version 3.6
# This code can run on other python 3 versions by removing sections of code that use print(f" words words {some variable}")
# This is found in energy function
# 1 in 10 test runs yields an incorrect number and I couldn't get it better than this, maybe some fault with my math.

step = 15


def seven_segment(pattern):

    def to_bool(a):
        if a==1:
            return True
        return False
    

    def hor(d):
        if d:
            print(" _ ")
        else:
            print("   ")
    
    def vert(d1,d2,d3):
        word=""

        if d1:
            word="|"
        else:
            word=" "
        
        if d3:
            word+="_"
        else:
            word+=" "
        
        if d2:
            word+="|"
        else:
            word+=" "
        
        print(word)

    

    pattern_b=list(map(to_bool,pattern))

    hor(pattern_b[0])
    vert(pattern_b[1],pattern_b[2],pattern_b[3])
    vert(pattern_b[4],pattern_b[5],pattern_b[6])

    number=0
    for i in range(0,4):
        if pattern_b[7+i]:
            number+=pow(2,i)
    print(int(number))


    

#initialse the weight matrix
def initialse_weight(p_l):

    weights = np.zeros(shape=(p_l,p_l))
    for i in range(0,p_l):
        for j in range(0,p_l):
            if i == j:
                weights[i][j] = 0
            else:
                weights[i][j] = random.uniform(-1, 1)
    return weights # random numbers

def combine(a,b,c):
    patterns = np.array([a,b,c])
    return patterns   # combine patterns to be learned into one array

def energyy(w, p_i, p_j):  #calculate energy of a given pattern and print energy
    
    piandw = np.dot(p_i,w)
    sum_pijwij = np.sum(np.dot(piandw,p_i))
    energy =  -0.5 * sum_pijwij

    print(f"Pattern energy = {energy}")
    return float(energy)
 

def learning(patterns):
    n = 1
    pats,neurons = patterns.shape               # get number of neurons and patterns
    weights = initialse_weight(neurons)         # initialise the weight matrix
    for p in patterns:
        weights = weights + np.outer(p,p)       # store patterns via weights, no neuron effects its own weight.
    weights[np.diag_indices(neurons)] = 0       # ensure all weight for i==j is 0 as fail safe
    weights = weights/pats                      # normalise weights

    for i in range (0,pats):
        print (f"########  Pattern, {n} ########")
        e = energyy(weights, patterns[i][:], patterns[i][:]) #print energy for each pattern
        n += 1

    return weights # train network  # train the network by adjusting the weight matrix which encodes patterns

def activ(x):

    if x < 0:
        x = -1
        return x
    else:
        x = 1
        return x
    #pattern[i] = int(pattern[i])
    #print (type(pattern)) #activation function to determine on or off status depending on sign of float
    


def recall(pattern, weights, step):  # attempt to recall pattern using weights and the input variables.
    
    #stepp = step
    n = 1
    energy = [100,50] # random values to begin while loop
    #for _ in range(0,stepp):
        #print("Iteration", n)
        #print(pattern)
        
        #pattern1 = (np.dot(pattern, weights))
        #pattern1 = pattern1.tolist()
        #pattern1 = list(map(activ, pattern1))
        #energy = energyy(weights, pattern, pattern1)
        #pattern = pattern1
        #n += 1      
    #  ^^^^ previous for loop based on steps not energy convergence

    #convergence checking
    while energy[-2] != energy[-1]:  # while the previous pattern energy is not equal to current energy itterate again
        print("Iteration", n)
        print(pattern)
        
        pattern1 = (np.dot(pattern, weights)) # synchronous updating by using matrix calculations rather than a for loop
        pattern1 = pattern1.tolist() # convert numpy array to list for operations
        pattern1 = list(map(activ, pattern1))  # convert map function to list for operations
        energy.extend([energyy(weights, pattern, pattern1)]) # add energy of current iteration to energy list to check for convergence at minima
        pattern = pattern1
        n += 1

    test = pattern
    return test


six = [1,1,-1,1,1,1,1,-1,1,1,-1]
three=[1,-1,1,1,-1,1,1,1,1,-1,-1]
one=[-1,-1,1,-1,-1,1,-1,1,-1,-1,-1]

patterns_mat = combine(six, three, one)
weights = learning(patterns_mat)

seven_segment(three)
seven_segment(six)
seven_segment(one)


print("######## Test 1 #########")

test=[1,-1,1,1,-1,1,1,-1,-1,-1,-1]
print(test)

test = recall(test, weights, step)

seven_segment(test)

#here the network should run printing at each step

print("######## Test 2 #########")

test=[1,1,1,1,1,1,1,-1,-1,-1,-1]
print(test)

test = recall(test, weights, step)

seven_segment(test)

#here the network should run printing at each step




# Some of the insight into this problem was gained using these websites

#  http://neuronaldynamics-exercises.readthedocs.io/en/latest/exercises/hopfield-network.html

# https://en.wikipedia.org/wiki/Hopfield_network

# http://codeaffectionate.blogspot.co.uk/2013/05/fun-with-hopfield-and-numpy.html
## this code gave me the idea to calculate the dot product as a way to do synchronous calculations.
