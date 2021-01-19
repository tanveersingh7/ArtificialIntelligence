'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 2
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 09 November, 2019

 mdpVI.py

 Value Iteration

'''

import numpy as np
import sys

reward = float(sys.argv[1])
print("Reward r is: ", reward)
gamma = 0.98

rows = [1,2,3]
columns = [1,2,3,4]

action_list = ["up","down","left","right"]
states = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]

reward_dict = {
    (1,1) : reward,
    (1,2) : reward,
    (1,3) : reward,
    (2,1) : reward,
    (2,2) : 0,
    (2,3) : reward,
    (3,1) : reward,
    (3,2) : reward,
    (3,3) : reward,
    (4,1) : reward,
    (4,2) : -1,
    (4,3) : 1
}

value_dict = {
    (1, 1): 0,
    (1, 2): 0,
    (1, 3): 0,
    (2, 1): 0,
    (2, 2): 0,
    (2, 3): 0,
    (3, 1): 0,
    (3, 2): 0,
    (3, 3): 0,
    (4, 1): 0,
    (4, 2): -1,
    (4, 3): 1
}

def next_state(state, action) :

    if action == "up" :
        new_state = (state[0], state[1]+1)
    elif action == "down" :
        new_state = (state[0], state[1]-1)
    elif action == "left" :
        new_state = (state[0]-1, state[1])
    elif action == "right" :
        new_state = (state[0]+1, state[1])


    if (new_state[0] >=1 and new_state[0] <=4) and (new_state[1]>=1 and new_state[1]<=3) :
        if new_state != (2,2) :
            return new_state

    return state

def action_output(state, action, values) :
    if action == "up" :
        ns1 = next_state(state, action)
        ns2 = next_state(state, "left")
        ns3 = next_state(state, "right")
    elif action == "left" :
        ns1 = next_state(state, action)
        ns2 = next_state(state, "up")
        ns3 = next_state(state, "down")
    elif action == "right" :
        ns1 = next_state(state, action)
        ns2 = next_state(state, "up")
        ns3 = next_state(state, "down")
    elif action == "down" :
        ns1 = next_state(state, action)
        ns2 = next_state(state, "left")
        ns3 = next_state(state, "right")

    output_of_action = 0.8 * values[ns1] + 0.1 * values[ns2] + 0.1 * values[ns3]

    return output_of_action


def value_iteration(states, value_dict, action_list, gamma, epsilon) :
    u1 = value_dict
    i = 0
    while True :
        i+=1
        u = dict(u1)
        delta = 0
        policy_dict = {}
        for state in states:
            if state == (2, 2) or state == (4,2) or state == (4,3):
                continue
            state_action_utility_dict = {}
            for action in action_list:
                utility_for_action = action_output(state, action, u)
                state_action_utility_dict[action] = utility_for_action
            #print("State action utility dict for state : ",state, " is  = ",state_action_utility_dict)
            array = sorted(state_action_utility_dict.items(), key=lambda x:x[1])
            output = array[-1][1]
            u1[state] = reward_dict[state] + gamma * output
            policy_dict[state] = array[-1][0]
            diff = abs(u1[state] - u[state])
            if diff > delta :
                delta = diff
        threshold = epsilon*(1-gamma)/gamma
        if delta < threshold :
            print("Policy for each state for reward = ", reward, " is  : ")
            for i in policy_dict:
                print(i, ":", policy_dict[i])
            return u1

epsilon =0.0001
new_value_dict = value_iteration(states,value_dict,action_list,gamma,epsilon)





