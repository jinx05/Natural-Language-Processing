import numpy as np
import pandas as pd

v = np.zeros(shape=(7, 5))
v_df = pd.DataFrame(v,columns=['Janet','will','back','the','bill'],index = ['NNP', 'MD','VB','JJ','NN','RB','DT'])
b = np.zeros(shape=(7, 5))
backpointer_df = pd.DataFrame(b,columns=['Janet','will','back','the','bill'],index = ['NNP', 'MD','VB','JJ','NN','RB','DT'])

# the transition probability matrix
A = np.array(
    [[0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026], [0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025],
     [0.0008, 0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041], [0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231],
     [0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036], [0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068],
     [0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479],
     [0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]])

A_df = pd.DataFrame(A,columns=['NNP', 'MD','VB','JJ','NN','RB','DT'],index=['pi','NNP', 'MD','VB','JJ','NN','RB','DT'])
# array converted to dataframe

O = np.array([[0.000032, 0, 0, 0.000048, 0], [0, 0.308431, 0, 0, 0], [0, 0.000028, 0.000672, 0, 0.000028],
              [0, 0, 0.000340, 0, 0], [0, 0.000200, 0.000223, 0, 0.002337], [0, 0, 0.010446, 0, 0],
              [0, 0, 0, 0.506099, 0]])
Ob_df = pd.DataFrame(O,columns=['Janet','will','back','the','bill'],index=['NNP', 'MD','VB','JJ','NN','RB','DT'])
# the observation matrix

print(" The observation matrix: ","\n",Ob_df)
print("The transition probability matrix: ","\n",A_df)

''' A is the state transition matrix, rows start form pi then NNP, MD,VB,JJ,NN,RB,DT and similar on the columns except the pi.
So, first I will fill the V matrix using the initialization step which is row 0 and then the dynamic programming formula takes over.'''

def getmax(v_df,A_df,Ob_df,next_state,step,input):
    max = 0
    max_st = ''
    for s in A_df.columns:
        n1 = A_df.loc[[s], [next_state]].values
        n2 = Ob_df.loc[[next_state], [input[step]]].values

        n = v_df.loc[[s], [input[step - 1]]].values * n1 * n2
        if n > max:
            max = n
            max_st = s
        else:
            continue
    # print(max_st)
    return max
# function to get the highest probability of reaching that state

# function gives which state gives the highest probability to the next state
def getbeststate(v_df,A_df,Ob_df,next_state,step,input):
    max = 0
    max_st = ''
    for s in A_df.columns:
        n1 = A_df.loc[[s], [next_state]].values[0][0]
        n2 = Ob_df.loc[[next_state], [input[step]]].values[0][0]

        n = v_df.loc[[s], [input[step - 1]]].values * n1 * n2
        if n > max:
            max = n
            max_st = s
        else:
            continue
    # print(max_st)
    return max_st
# function to get argmax

def get_tag(v_df,input,a_df):
    a = np.array(v_df[input])
    i = np.argmax(a)
    return a_df.columns[i]
# for the last word the argmax on the last column

''' The viterbi algorithm as given in the slides'''
def viterbi(A_df,v_df,input,Ob_df,backpointer_df):
    for i in A_df.columns:
        n1 = A_df.loc[['pi'], [i]].values[0][0]
        # print(type(n1))
        n2 = Ob_df.loc[[i], [input[0]]].values[0][0]
        # print(n2)
        v_df.loc[[i], [input[0]]] = n1 * n2
        backpointer_df.loc[[i], [input[0]]] = 0
# initialisation
    for next in range(1, len(input)):
        for state in A_df.columns:
            v_df.loc[[state], [input[next]]] = getmax(v_df, A_df, Ob_df, state, next,input)
            backpointer_df.loc[[state], [input[next]]] = getbeststate(v_df, A_df, Ob_df, state, next,input)

    print("The viterbi matrix: ","\n",v_df)
    print("The backpointer matrix: ","\n",backpointer_df)
    return v_df, backpointer_df

def getallthetags(back_df,last_tag,input):
    final_tags = []
    final_tags.append(str(last_tag))
# traversing the backpointer to get all the states
    for k in range(len(input),1,-1):
        # we don't need to go to the last value as the backpointer stores the previous tag(state).
        n1 = input[k-1]
        n2 = final_tags[-1]
        ptag = back_df.loc[[n2],[n1]].values[0][0]
        final_tags.append(ptag)
    return final_tags

inputs = ['Janet will back the bill','will Janet back the bill','back the bill Janet will']

for inp in inputs:
    inp1 = inp.split(",")
    for question in inp1 :
        input = question.split()

        V, B = viterbi(A_df, v_df, input, Ob_df, backpointer_df)

        last_tag = get_tag(V, input[-1], A_df)

        output = getallthetags(B, last_tag, input)
        output = output[::-1]
        print('******************** program output *************')
        print("\n")
        print("For the given Input: ",input,"The Most likely Tag sequence is: ", output)
        print("The porbability for the resulting sequence of tags: ",V.loc[[last_tag],[input[-1]]].values[0][0])
        print("\n")


