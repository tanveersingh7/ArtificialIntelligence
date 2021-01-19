'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 2
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 09 November, 2019

 mdpPI.py

 Policy Iteration

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

initial_policy_dict = {
    (1, 1): "up",
    (1, 2): "up",
    (1, 3): "up",
    (2, 1): "up",
    (2, 3): "up",
    (3, 1): "up",
    (3, 2): "up",
    (3, 3): "up",
    (4, 1): "up",
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

def policy_evaluation(states, values, policy_dict, gamma) :
    new_u = values
    for i in range(20) :
        u = dict(new_u)
        for state in states :
            if state == (2, 2) or state == (4, 2) or state == (4, 3):
                continue
            output = action_output(state, policy_dict[state], u)
            new_u[state] = reward_dict[state] + gamma * output
    return new_u



def policy_iteration(states, value_dict, policy_dict, action_list, gamma) :
    u1 = value_dict
    policy = policy_dict
    print("Initial policy : ", policy)
    i = 0
    while True:
        i += 1
        new_policy = dict(policy)
        new_u1 = policy_evaluation(states, u1, new_policy, gamma)
        unchanged = True
        for state in states:
            if state == (2, 2) or state == (4, 2) or state == (4, 3):
                continue
            state_action_utility_dict = {}
            for action in action_list:
                utility_for_action = action_output(state, action, new_u1)
                state_action_utility_dict[action] = utility_for_action
            # print("State action utility dict for state : ",state, " is  = ",state_action_utility_dict)
            array = sorted(state_action_utility_dict.items(), key=lambda x: x[1])
            max_utility_output = array[-1][1]
            utility_output_policy = action_output(state, policy[state], new_u1)
            if(max_utility_output > utility_output_policy) :
                policy[state] = array[-1][0]
                unchanged = False
        u1 = new_u1
        if unchanged == True :
            return policy


new_policy_dict = policy_iteration(states, value_dict, initial_policy_dict, action_list,gamma)
print("Policy for each state for reward = ", reward, "after policy iteration is  : ")
for i in new_policy_dict:
    print(i, ":", new_policy_dict[i])