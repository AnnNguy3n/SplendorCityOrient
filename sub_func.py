import numpy as np
from numba import njit

normalCardInfor = np.array([[0, 2, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 2, 1, 1, 0, 2, 1], [0, 2, 0, 1, 0, 0, 2], [0, 2, 0, 3, 1, 0, 1], [0, 2, 1, 1, 0, 1, 1], [1, 2, 0, 0, 0, 4, 0], [0, 2, 2, 1, 0, 2, 0], [0, 1, 2, 0, 2, 0, 1], [0, 1, 0, 0, 2, 2, 0], [0, 1, 1, 0, 1, 1, 1], [0, 1, 2, 0, 1, 1, 1], [0, 1, 1, 1, 3, 0, 0], [0, 1, 0, 0, 0, 2, 1], [0, 1, 0, 0, 0, 3, 0], [1, 1, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 1, 2, 2], [0, 0, 1, 0, 0, 3, 1], [0, 0, 2, 0, 0, 0, 2], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 2, 1, 0, 0], [0, 4, 0, 2, 2, 1, 0], [0, 4, 1, 1, 2, 1, 0], [0, 4, 0, 1, 0, 1, 3], [1, 4, 0, 0, 4, 0, 0], [0, 4, 0, 2, 0, 2, 0], [0, 4, 2, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 0], [0, 4, 0, 3, 0, 0, 0], [0, 3, 1, 0, 2, 0, 0], [0, 3, 1, 1, 1, 0, 1], [1, 3, 0, 4, 0, 0, 0], [0, 3, 1, 2, 0, 0, 2], [0, 3, 0, 0, 3, 0, 0], [0, 3, 0, 0, 2, 0, 2], [0, 3, 3, 0, 1, 1, 0], [0, 3, 1, 2, 1, 0, 1], [1, 2, 0, 3, 0, 2, 2], [2, 2, 0, 2, 0, 1, 4], [1, 2, 3, 0, 2, 0, 3], [2, 2, 0, 5, 3, 0, 0], [2, 2, 0, 0, 5, 0, 0], [3, 2, 0, 0, 6, 0, 0], [3, 1, 0, 6, 0, 0, 0], [2, 1, 1, 0, 0, 4, 2], [2, 1, 0, 5, 0, 0, 0], [2, 1, 0, 3, 0, 0, 5], [1, 1, 0, 2, 3, 3, 0], [1, 1, 3, 2, 2, 0, 0], [3, 0, 6, 0, 0, 0, 0], [2, 0, 0, 0, 0, 5, 3], [2, 0, 0, 0, 0, 5, 0], [2, 0, 0, 4, 2, 0, 1], [1, 0, 2, 3, 0, 3, 0], [1, 0, 2, 0, 0, 3, 2], [3, 4, 0, 0, 0, 0, 6], [2, 4, 5, 0, 0, 3, 0], [2, 4, 5, 0, 0, 0, 0], [1, 4, 3, 3, 0, 0, 2], [1, 4, 2, 0, 3, 2, 0], [2, 4, 4, 0, 1, 2, 0], [1, 3, 0, 2, 2, 0, 3], [1, 3, 0, 0, 3, 2, 3], [2, 3, 2, 1, 4, 0, 0], [2, 3, 3, 0, 5, 0, 0], [2, 3, 0, 0, 0, 0, 5], [3, 3, 0, 0, 0, 6, 0], [4, 2, 0, 7, 0, 0, 0], [4, 2, 0, 6, 3, 0, 3], [5, 2, 0, 7, 3, 0, 0], [3, 2, 3, 3, 0, 3, 5], [3, 1, 3, 0, 3, 5, 3], [4, 1, 0, 0, 0, 0, 7], [5, 1, 0, 3, 0, 0, 7], [4, 1, 0, 3, 0, 3, 6], [3, 0, 0, 5, 3, 3, 3], [4, 0, 0, 0, 7, 0, 0], [5, 0, 3, 0, 7, 0, 0], [4, 0, 3, 3, 6, 0, 0], [5, 4, 0, 0, 0, 7, 3], [3, 4, 5, 3, 3, 3, 0], [4, 4, 0, 0, 0, 7, 0], [4, 4, 3, 0, 0, 6, 3], [3, 3, 3, 3, 5, 0, 3], [5, 3, 7, 0, 0, 3, 0], [4, 3, 6, 0, 3, 3, 0], [4, 3, 7, 0, 0, 0, 0]])
nobleCardInfor = np.array([[3, 0, 4, 4, 0, 0], [3, 3, 0, 3, 3, 0], [3, 3, 3, 3, 0, 0], [3, 3, 0, 0, 3, 3], [3, 0, 3, 0, 3, 3], [3, 4, 0, 4, 0, 0], [3, 4, 0, 0, 4, 0], [3, 0, 3, 3, 0, 3], [3, 0, 4, 0, 0, 4], [3, 0, 0, 0, 4, 4]])
orientCardInfor = np.array([[0, 6, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0], [0, 6, 1, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [0, 6, 1, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0], [0, 6, 1, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0], [0, 6, 1, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0], [0, 3, 2, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 2, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0], [0, 6, 1, 1, 0, 0, 3, 0, 1, 4, 0, 0, 0, 0, 0], [0, 6, 1, 1, 0, 3, 0, 4, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0], [1, 2, 1, 0, 1, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0], [1, 4, 1, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 0, 1, 0, 0, 6, 3, 0, 0, 0, 0, 0], [0, 1, 1, 2, 0, 3, 1, 6, 0, 0, 0, 0, 0, 0, 0], [0, 2, 1, 2, 0, 6, 0, 1, 3, 0, 0, 0, 0, 0, 0], [0, 3, 1, 2, 0, 0, 3, 0, 1, 6, 0, 0, 0, 0, 0], [0, 4, 1, 2, 0, 0, 6, 3, 0, 1, 0, 0, 0, 0, 0], [3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], [3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], [3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], [3, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]])


@njit
def fillCard(posE, cardId, env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3):
    if cardId < 40:
        if lv1[-1] < 40:
            env[posE] = lv1[lv1[-1]]
            lv1[-1] += 1
        else:
            env[posE] = -1
    elif cardId < 70:
        if lv2[-1] < 30:
            env[posE] = lv2[lv2[-1]]
            lv2[-1] += 1
        else:
            env[posE] = -1
    elif cardId < 90:
        if lv3[-1] < 20:
            env[posE] = lv3[lv3[-1]]
            lv3[-1] += 1
        else:
            env[posE] = -1
    elif cardId < 100:
        if oriLv1[-1] < 10:
            env[posE] = oriLv1[oriLv1[-1]]
            oriLv1[-1] += 1
        else:
            env[posE] = -1
    elif cardId < 110:
        if oriLv2[-1] < 10:
            env[posE] = oriLv2[oriLv2[-1]]
            oriLv2[-1] += 1
        else:
            env[posE] = -1
    elif cardId < 120:
        if oriLv3[-1] < 10:
            env[posE] = oriLv3[oriLv3[-1]]
            oriLv3[-1] += 1
        else:
            env[posE] = -1


@njit
def checkNoble(env, tempVal, pPerStocks):
    for i in range(5):
        pos = 6+i
        if env[pos] != -1:
            price = nobleCardInfor[env[pos]][1:6]
            if (pPerStocks >= price).all():
                env[pos] = -1
                env[41+tempVal] += 3
    
    for i in range(3):
        temp_ = 45+tempVal+i
        if env[temp_] != -1:
            price = nobleCardInfor[env[temp_]][1:6]
            if (pPerStocks >= price).all():
                env[temp_] = -1
                env[41+tempVal] += 3


@njit
def checkBuyBaseCard(gems, perGems, price, orientGoldToken):
    if np.sum((price>(gems[0:5]+perGems))*(price-gems[0:5]-perGems)) <= gems[5] + orientGoldToken:
        return True

    return False


@njit
def checkBuyOrientCard(gems, perGems, gemPrice, perGemPrice, orientGoldToken):
    if (gemPrice > 0).any():
        if np.sum((gemPrice>(gems[0:5]+perGems))*(gemPrice-gems[0:5]-perGems)) <= gems[5] + orientGoldToken:
            return True
        
        return False
    
    if (perGemPrice > 0).any():
        if (perGemPrice > perGems).any():
            return False

        return True


@njit
def convertNoble(sIdxState, sIdxEnv, pos, env, state):
    nobleIdx = env[sIdxEnv+pos]
    if nobleIdx != -1:
        tempVal = 6*pos
        state[sIdxState+tempVal:sIdxState+6+tempVal] = nobleCardInfor[nobleIdx]


@njit
def convertBase(sIdxState, sIdxEnv, pos, env, state):
    cardId = env[sIdxEnv+pos]
    if cardId != -1:
        cardInfor = normalCardInfor[cardId]
        tempVal = 11*pos
        state[sIdxState+tempVal] = cardInfor[0]
        state[sIdxState+1+cardInfor[1]+tempVal] = 1
        state[sIdxState+6+tempVal:sIdxState+11+tempVal] = cardInfor[2:7]


@njit
def convertOrient(sIdxState, sIdxEnv, pos, env, state):
    cardId = env[sIdxEnv+pos] - 90
    if cardId != -91:
        cardInfor = orientCardInfor[cardId]
        tempVal = 21*pos
        state[sIdxState+tempVal] = cardInfor[0]
        state[sIdxState+1+cardInfor[1]+tempVal] = cardInfor[2]
        if cardInfor[3] != 0:
            state[sIdxState+7+cardInfor[3]+tempVal] = 1
        
        state[sIdxState+10+tempVal] = cardInfor[4]
        state[sIdxState+11+tempVal:sIdxState+16+tempVal] = cardInfor[5:10]
        state[sIdxState+16+tempVal:sIdxState+21+tempVal] = cardInfor[10:15]
