from env import *
from time import time

perFile = List()
perFile.append(np.array([[0., 0.]]))

@njit
def numbaRandomBot1(state, tempData, perData):
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    if getReward(state) == 1:
        perData[0][0][1] += 1
    if getReward(state) == -1:
        perData[0][0][0] += 1
    return validActions[idx], tempData, perData

t_ = time()
numbaMain(numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,numbaRandomBot1, 10000, perFile, True, 1000)
print(time()-t_, perFile)


