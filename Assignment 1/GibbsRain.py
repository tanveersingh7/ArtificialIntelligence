'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 1
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 21 September, 2019

 GibbsRain.py


USING THE CONDITIONAL PROBABILITIES FOR Cloudy, Sprinkler, Wet Grass and Rain,
WE ESTIMATE P(r|s,w) USING GIBBS SAMPLING FOR 100 AND 10000 STEPS.

GIBBS SAMPLING ALGORITHM:
	1. Construct a Markov Chain based on the given Bayesian Network.
    2. Start with a random state for the random variables.Fix the evidence
    variables at their observed values.
    3. Generate the next state by randomly sampling a value for one of the non
    evidence variables.Sampling for a variable(Xi) is done depending on the
    current values of the variables in the Markov Blanket of Xi.
    4. Sampling each variable in turn keeping the evidence variables fixed.
'''
#Importing the necessary libraries
import numpy as np
import sys

#Number of steps provided as an argumen to the program
numSteps = int(sys.argv[1])
print('Number of steps for Gibbs sampling: ',numSteps)

#The non environment variables for which we need to sample
non_ev_variables= np.array(['Cloudy','Rain'])

#Random variables initialised to their initial state.
initial_states = {
    'Cloudy' : True,
    'Rain' : False,
    'Sprinkler' : True,
    'Wet Grass' : True
}
#Dictionary called 'current states' maintains the state of the random variables
current_states = initial_states
print("Initial states of the random variables are :" )
print(initial_states)

#Dictionary called 'probability_dict' maintains the conditional probabilities of the non evidence variables w.r.t other
#random variables. THe evidence variables 'Sprinkler' and 'Wet Grass' are fixed to True
probability_dict = {
    'Cloudy' : {
        'Rain' : 0.444,
        '~Rain' : 0.047
    },
    '~Cloudy' : {
        'Rain' : 0.555,
        '~Rain' : 0.952

    },
    'Rain' : {
        'Cloudy' : 0.814,
        '~Cloudy' : 0.215
    },
    '~Rain' : {
        'Cloudy' : 0.185,
        '~Cloudy' : 0.784
    }
}


#Counters maintaining the true count and false count for evidence variable 'Rain'
true_count = 0
false_count = 0


def GibbsSampler(value_ne_1,value_ne_2) :
    probability = probability_dict[value_ne_1][value_ne_2]
    probability_complement = 1 - probability

    sample_value = np.random.choice([True,False], p = [probability, probability_complement])

    return sample_value

i=0
for i in range(int(numSteps)) :

#    print("Iteration  number: ",i+1)

    n_ev_v1 = non_ev_variables[i%2]
    n_ev_v2 = non_ev_variables[(i+1)%2]
    
    n_ev_v1_value = n_ev_v1
    n_ev_v2_value = n_ev_v2
    
    state_ne_1 = current_states[n_ev_v1]
    state_ne_2 = current_states[n_ev_v2]
    
    if (state_ne_1 == False) :
        n_ev_v1_value = '~' + n_ev_v1
    
    if (state_ne_2 == False) :
        n_ev_v2_value = '~' + n_ev_v2

    sample_value = GibbsSampler(n_ev_v1_value, n_ev_v2_value)

#    print("sample_value is :", sample_value)

    if(sample_value == False) :
        if( n_ev_v1_value == 'Rain' or n_ev_v1_value == 'Cloudy') :
            current_states[n_ev_v1] = sample_value
            if( n_ev_v1_value == 'Rain') :
                false_count += 1
        
        if( n_ev_v1_value == '~Rain' or n_ev_v1_value == '~Cloudy') :
            current_states[n_ev_v1] = not sample_value
            if( n_ev_v1_value == '~Rain') :
                true_count += 1
    
    if(sample_value == True) :
        if( n_ev_v1_value == 'Rain' or n_ev_v1_value == 'Cloudy') :
            current_states[n_ev_v1] = sample_value
            if( n_ev_v1_value == 'Rain') :
                true_count += 1
        
        if( n_ev_v1_value == '~Rain' or n_ev_v1_value == '~Cloudy') :
            current_states[n_ev_v1] = not sample_value
            if( n_ev_v1_value == '~Rain') :
                false_count += 1

#    print("State of random variables after ",i+1,"  iterations of Gibbs sampling:")

print("True count :", true_count)  

print("False count :", false_count)  

print("Final States after " , i+1, "iterations :")
print(current_states)

print("Probability that (Rain = true | Sprinkler = true, WetGrass = true) is :",true_count/numSteps)





    


    
