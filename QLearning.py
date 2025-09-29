# -----------------------------------------------------------------
# Game Day 1:  Learning Day
# QLearning algorithm implemented by Leen-Kiat Soh
#
# 1.  The program below is not documented so that each team's 
# responsibility is to document the following code clearly 
# and completely.
#
# 2.  The function computeAlpha() is implemented as a default design.
# Each team's responsiblity is to revise the design according to their
# game strategy.
#
# 3.  The function decideAction() is implemented as a random action
# selector.  Each team's responsiblity is to revise the design 
# according to their game strategy.
#
# Note also that the program reads in two input csv files so as to 
# simulate the transition probability of (s,a,s') and generate the 
# rewards r(s,a,s').  With these two functions, the main body of the 
# program essentially receives the next state from the "environment" 
# together with the next state's rewards.
# 
# You are not allowed to modify the code after "MAIN BODY" except for
# one line regarding the alpha (i.e., learning rate).
# -----------------------------------------------------------------

import random
import numpy as np
import csv
# import matplotlib.pyplot as plt

MAX_ITERATION = 250000
ALPHA = 0.5
BETA = 0.9
EPSILON = 0.1

qTable = []
tpTable = []

MAX_NUM_ACTIONS = 6
MAX_NUM_STATES = 6

def initialize():
   # initialize the Q table and the count of state-action pairs
   for i in range(0,MAX_NUM_STATES):
      temp = []
      for j in range(0,MAX_NUM_ACTIONS):
         temp.append(0)
      # end for j
      qTable.append(temp)
   # end for i

# end initialize

def readTPTable():

   # first initialize
   for i in range(0,MAX_NUM_STATES):
      temp = []
      for j in range(0,MAX_NUM_ACTIONS):
         temp2 = []
         for k in range(0,MAX_NUM_STATES):
            temp2.append(0)
	 # end for k
         temp.append(temp2)
      # end for j
      tpTable.append(temp)
   # end for j

   # then read from input file
   with open("TPTable.csv",'r') as csv_infile:
      data_reader = csv.reader(csv_infile,delimiter=',')
      for row in data_reader:
         s = int(row[0])
         a = int(row[1])
         s_next = int(row[2])
         p = float(row[3])
         tpTable[s][a][s_next] = p
      # end for
   # end with open

# end readTPTable      

def obtainReward(s,a,s_next):

   rTable = []
   with open("r.csv",'r') as csv_infile:
      data_reader = csv.reader(csv_infile,delimiter=',')
      for row in data_reader:
         rTable.append(float(row[0]))
      # end for
   # end with open

   r = np.random.normal(rTable[s_next], 0.1, 1)

   return r

# end obtainReward

def obtainNextState(s,a):

   pnum = random.random()
   i = 0
   lb = 0
   found = False

   while (found == False and i < MAX_NUM_STATES):
      ub = lb + tpTable[s][a][i]
      if (pnum >= lb and pnum <= ub):
         found = True
      else:
         lb = ub
         i = i + 1
      # end if
   # end while

   if i >= MAX_NUM_STATES:
      i = s # back to the current state

   return i

# end obtainNextState

def cosin_decay_scheduler(hyperparam, t):
    """Cosine decay scheduler.
    hyperparam: initial value of the hyperparameter (it could be learning rate or epsilon)
    t: current step
    # 0 < t1 < t2 < MAX_ITERATION
    """
    t1  = 50_000
    t2 = 200_000
    max_value = hyperparam
    min_value = max_value * 0.1
    if t < t1:
        return max_value * (t+1) / t1
    elif t > t2:
        return min_value
    
    # in between, using cosin decay
    decay_ratio = (t - t1) / (t2 - t1)
    coeff = 0.5 * (1 + np.cos(np.pi * decay_ratio))
    return min_value + coeff * (max_value - min_value)


def computeAlpha(alpha, t):
    return cosin_decay_scheduler(alpha, t)


def computeEpsilon(epsilon, t):
    return cosin_decay_scheduler(epsilon, t)
    

def decideAction(s, t):
   epsilon = computeEpsilon(EPSILON, t)
   if random.random() < epsilon:
       return random.randint(0,MAX_NUM_ACTIONS-1)
   # end if
   else:
      max_value = max(qTable[s])
      best_actions = [i for i in range(MAX_NUM_ACTIONS) if qTable[s][i] == max_value]
      return random.choice(best_actions)
# end decideAction


def computeValue(s):
   max = qTable[s][0]
   for j in range(1,MAX_NUM_ACTIONS):
      if (qTable[s][j] > max):
         max = qTable[s][j]
   # end for j
   return max
# end computeValue

# -------------------------------------------------------
# MAIN BODY
# -------------------------------------------------------
# Do not modify the code in the following section, 
# except for the line that dictates how alpha 
# will change or stay as a constant.
# -------------------------------------------------------

initialize()
readTPTable()
t = 0
s = 0
rewardSoFar = 0
it_lrs = []

while (t < MAX_ITERATION):
   alpha = computeAlpha(ALPHA, t) # only if we want alpha to be dependent on t
   # it_lrs.append(alpha)
   a = decideAction(s, t) 
   s_next = obtainNextState(s,a)
   if (s == s_next):
      r = 0
   else:
      r = obtainReward(s,a,s_next)
   rewardSoFar = rewardSoFar + r
   val = computeValue(s_next)
   qTable[s][a] = (1 - alpha)*qTable[s][a] + alpha*(r + BETA*val)
   s = s_next
   t = t + 1
  
# end while

# print out the Q table
for i in range(0,MAX_NUM_STATES):
   for j in range(0,MAX_NUM_ACTIONS):
      print("["+str(i)+"]["+str(j)+"]: "+str(qTable[i][j]))

# print out final total rewards
print("Total rewards = " + str(rewardSoFar))

# its = [i for i in range(MAX_ITERATION)]
# # plot learning rate schedule
# try:
#    # Try to use an ASCII/terminal plotting library first so the plot can be
#    # rendered directly in a terminal (no GUI required).
#    import plotext as pxt

#    # plotext expects lists of numbers; these are already lists
#    pxt.plot(its, it_lrs, color='blue')
#    pxt.title('Learning Rate Schedule')
#    pxt.xlabel('Iteration')
#    pxt.ylabel('Learning Rate')
#    pxt.show()
# except Exception:
#    # Fallback: save the matplotlib figure to a PNG file
#    plt.plot(its, it_lrs, color='blue')
#    plt.xlabel('Iteration')
#    plt.ylabel('Learning Rate')
#    plt.title('Learning Rate Schedule')
#    outfn = "learning_rate.png"
#    plt.savefig(outfn)
#    print("Plot saved to {}. Open it with an image viewer if your terminal doesn't show images.".format(outfn))
