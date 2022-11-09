from env import *
from time import time

perFile = List()
perFile.append(np.array([[0.]]))

@njit
def numbaRandomBot1(state, tempData, perData):
    perData[0][0][0] += 1
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    return validActions[idx], tempData, perData

t_ = time()
numbaMain(numbaRandomBot1,numbaRandomBot1,numbaRandomBot1,numbaRandomBot1, 10000, perFile, True, 1000)
print(time()-t_, perFile)


