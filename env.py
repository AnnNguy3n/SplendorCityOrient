from sub_func import *
from numba.typed import List
from numba import jit

__ENV_SIZE__ = 126
__STATE_SIZE__ = 552
__ACTION_SIZE__ = 82
__AGENT_SIZE__ = 4


@njit
def initEnv():
    lv1 = np.arange(41) # Khởi tạo chồng thẻ ẩn cấp 1
    lv2 = np.arange(40, 71) # Khởi tạo chồng thẻ ẩn cấp 2
    lv3 = np.arange(70, 91) # Khởi tạo chồng thẻ ẩn cấp 3
    oriLv1 = np.arange(90, 101) # Chồng thẻ orient cấp 1
    oriLv2 = np.arange(100, 111) # Chồng thẻ orient cấp 2
    oriLv3 = np.arange(110, 121) # Chồng thẻ orient cấp 3

    np.random.shuffle(lv1[:-1]) # Xáo trộn chồng thẻ ẩn cấp 1
    np.random.shuffle(lv2[:-1]) # Xáo trộn chồng thẻ ẩn cấp 2
    np.random.shuffle(lv3[:-1]) # Xáo trộn chồng thẻ ẩn cấp 3
    np.random.shuffle(oriLv1[:-1]) # Xáo trộn chồng thẻ Orient ẩn cấp 1
    np.random.shuffle(oriLv2[:-1]) # Xáo trộn chồng thẻ Orient ẩn cấp 2
    np.random.shuffle(oriLv3[:-1]) # Xáo trộn chồng thẻ Orient ẩn cấp 3

    lv1[-1] = 4 # Số thẻ cấp 1 chia ra bàn chơi: 4
    lv2[-1] = 4 # Số thẻ cấp 2 chia ra bàn chơi: 4
    lv3[-1] = 4 # Số thẻ cấp 3 chia ra bàn chơi: 4
    oriLv1[-1] = 2 # Số thẻ Orient cấp 1 chia ra bàn chơi: 4
    oriLv2[-1] = 2 # Số thẻ Orient cấp 2 chia ra bàn chơi: 4
    oriLv3[-1] = 2 # Số thẻ Orient cấp 3 chia ra bàn chơi: 4

    env = np.full(__ENV_SIZE__, 0)

    env[0:5] = 7 # Nguyên liệu thường
    env[5] = 5 # Nguyên liệu vàng

    noble = np.arange(10) # Khởi tạo chồng thẻ quý tộc
    np.random.shuffle(noble) # Xáo trộn
    env[6:11] = noble[:5] # Lấy 5 thẻ noble và xếp trên bàn chơi

    env[11:15] = lv1[:4] # Lấy 4 thẻ lv1 và xếp trên bàn chơi
    env[15:19] = lv2[:4] # Lấy 4 thẻ lv2 và xếp trên bàn chơi
    env[19:23] = lv3[:4] # Lấy 4 thẻ lv3 và xếp trên bàn chơi

    env[23:25] = oriLv1[:2] # Lấy 2 thẻ Orient lv1 và xếp trên bàn chơi
    env[25:27] = oriLv2[:2] # Lấy 2 thẻ Orient lv2 và xếp trên bàn chơi
    env[27:29] = oriLv3[:2] # Lấy 2 thẻ Orient lv3 và xếp trên bàn chơi

    for pIdx in range(4): # Thông tin của các người chơi [29:48:67:86:105]
        tempVal = 19*pIdx
        # env[29+tempVal:42+tempVal] = 0
        env[42+tempVal:48+tempVal] = -1

    # env[105] = 0 # Turn
    # env[106] = 0 # Phase
    # env[107:112] = 0 # Nguyên liệu người chơi đã lấy trong turn đó
    # env[112:116] = 0 # Số thẻ đã mua của các người chơi
    # env[116] = 0 # Game đã kết thúc hay chưa (1 là kết thúc rồi)
    # env[117] = 0 # Được lấy thẻ free cấp mấy

    # env[118:123] = 0 # Special Case phase 3
    # env[123] = 0 # Special Case phase 3
    # env[124] = 0 # Special Case phase 3
    env[125] = -1 # Special Case phase 3

    return env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3



def visualizeEnv(env_, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3):
    env = env_.copy()
    dict_ = {}
    dict_['BoardGems'] = env[0:6]
    dict_['NobleID'] = env[6:11] + 1

    dict_['BaseCard'] = {}
    dict_['BaseCard']['Lv1'] = env[11:15] + 1
    dict_['BaseCard']['Lv2'] = env[15:19] + 1
    dict_['BaseCard']['Lv3'] = env[19:23] + 1

    dict_['OrientCard'] = {}
    dict_['OrientCard']['Lv1'] = env[23:25] + 1
    dict_['OrientCard']['Lv2'] = env[25:27] + 1
    dict_['OrientCard']['Lv3'] = env[27:29] + 1

    for i in range(4):
        dict_[f'Player_{i}'] = {}
        pInfor = env[29+19*i:48+19*i]
        dict_[f'Player_{i}']['Gems'] = pInfor[0:6]
        dict_[f'Player_{i}']['PerGems'] = pInfor[6:11]
        dict_[f'Player_{i}']['OrientGoldGems'] = pInfor[11]
        dict_[f'Player_{i}']['Score'] = pInfor[12]
        dict_[f'Player_{i}']['HidingCards'] = pInfor[13:16] + 1
        dict_[f'Player_{i}']['HidingNobles'] = pInfor[16:19] + 1
    
    dict_['Turn'] = env[105]
    dict_['Phase'] = env[106]
    dict_['TakenStocks'] = env[107:112]
    dict_['NumBoughtCards'] = env[112:116]
    dict_['EndGame'] = env[116]
    dict_['HideCardsLv1Order'] = lv1[:-1], lv1[-1]
    dict_['HideCardsLv2Order'] = lv2[:-1], lv2[-1]
    dict_['HideCardsLv3Order'] = lv3[:-1], lv3[-1]
    dict_['HideOrientLv1Order'] = lv1[:-1], lv1[-1]
    dict_['HideOrientLv2Order'] = lv2[:-1], lv2[-1]
    dict_['HideOrientLv3Order'] = lv3[:-1], lv3[-1]
    
    return dict_


@njit
def getStateSize():
    return __STATE_SIZE__


@njit
def getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3):
    state = np.zeros(__STATE_SIZE__)

    state[0:6] = env[0:6]

    for i in range(5):
        convertNoble(6, 6, i, env, state)
    
    for i in range(12):
        convertBase(36, 11, i, env, state)
    
    for i in range(6):
        convertOrient(168, 23, i, env, state)
    
    if lv1[-1] < 40: # Còn thẻ ẩn cấp 1
        state[294] = 1
    if lv2[-1] < 30: # Còn thẻ ẩn cấp 2
        state[295] = 1
    if lv3[-1] < 20: # Còn thẻ ẩn cấp 3
        state[296] = 1
    if oriLv1[-1] < 10: # Còn thẻ ẩn cấp 1
        state[297] = 1
    if oriLv2[-1] < 10: # Còn thẻ ẩn cấp 2
        state[298] = 1
    if oriLv3[-1] < 10: # Còn thẻ ẩn cấp 3
        state[299] = 1
    
    pIdx = env[105] % 4
    for i in range(4):
        pEnvIdx = (pIdx + i) % 4
        tempVal = 13*i
        tempValEnv = 19*pEnvIdx
        state[300+tempVal:313+tempVal] = env[29+tempValEnv:42+tempValEnv]
        if i == 0:
            tempValCard = 42+tempValEnv
            for pos in range(3):
                cardId = env[tempValCard+pos]
                if cardId != -1:
                    if cardId < 90:
                        convertBase(352, tempValCard, pos, env, state)
                    else:
                        convertOrient(385, tempValCard, pos, env, state)
            
            tempValCard = 45+tempValEnv
            for pos in range(3):
                convertNoble(502, tempValCard, pos, env, state)
        else:
            tempValP = 9*(i - 1)
            tempValCard = 42+tempValEnv
            for pos in range(3):
                cardId = env[tempValCard+pos]
                if cardId != -1:
                    if cardId < 90:
                        tempValCardLv = 3*pos
                        if cardId < 40:
                            state[448+tempValP+tempValCardLv] = 1
                        elif cardId < 70:
                            state[449+tempValP+tempValCardLv] = 1
                        elif cardId < 90:
                            state[450+tempValP+tempValCardLv] = 1
                    else:
                        cardId = cardId - 90
                        tempValCardLv = 3*pos
                        if cardId < 10:
                            state[475+tempValP+tempValCardLv] = 1
                        elif cardId < 20:
                            state[476+tempValP+tempValCardLv] = 1
                        elif cardId < 30:
                            state[477+tempValP+tempValCardLv] = 1
            
            state[519+i] = np.count_nonzero(env[45+tempValEnv:48+tempValEnv] != -1)
        
        state[540+i] = env[112+pIdx]
    
    state[523+pIdx] = 1

    state[527+env[106]] = 1

    state[535:540] = env[107:112]

    state[544] = env[116]

    if env[117] != 0:
        state[544+env[117]] = 1
    
    state[547:552] = env[118:123]

    return state


@njit
def getActionSize():
    return __ACTION_SIZE__


@njit
def getValidActions(state):
    phase = state[527:535]
    validActions = np.full(__ACTION_SIZE__, 0)

    if phase[0] == 1: # Lựa chọn kiểu hành động
        boardStocks = state[0:6]
        if (boardStocks[0:5] > 0).any(): # Bàn chơi còn nguyên liệu thường
            validActions[1] = 1 # Lấy nguyên liệu
        else: # Bàn chơi hết nguyên liệu thường
            validActions[0] = 1 # Bỏ lượt
        
        # Check số thẻ úp của bản thân
        countReserveCard = 0

        for i in range(3): # Thẻ thường
            tempVal = 11*i
            if (state[358+tempVal:363+tempVal] > 0).any():
                countReserveCard += 1
        
        for i in range(3): # Thẻ orient
            tempVal = 21*i
            if (state[396+tempVal:406+tempVal] > 0).any():
                countReserveCard += 1
        
        if countReserveCard < 3:
            validActions[2] = 1
        
        # Check action mua thẻ
        for i in range(12): # Check thẻ thường
            tempVal = 11*i
            cardPrice = state[42+tempVal:47+tempVal]
            if (cardPrice > 0).any() and checkBuyBaseCard(state[300:306], state[306:311], cardPrice, state[311]):
                validActions[3] = 1
                break
        else:
            for i in range(6): # Check thẻ Orient
                tempVal = 21*i
                cardPrice = state[179+tempVal:184+tempVal]
                cardPerPrice = state[184+tempVal:189+tempVal]
                if ((cardPrice > 0).any() or (cardPerPrice > 0).any()) and checkBuyOrientCard(state[300:306], state[306:311], cardPrice, cardPerPrice, state[311]):
                    if state[175+tempVal] == 0 or (state[306:311] > 0).any():
                        validActions[3] = 1
                        break
            else:
                for i in range(3): # Check 3 thẻ thường đang úp
                    tempVal = 11*i
                    cardPrice = state[358+tempVal:363+tempVal]
                    if (cardPrice > 0).any() and checkBuyBaseCard(state[300:306], state[306:311], cardPrice, state[311]):
                        validActions[3] = 1
                        break
                else:
                    for i in range(3): # Check 3 thẻ Orient đang úp
                        tempVal = 21*i
                        cardPrice = state[396+tempVal:401+tempVal]
                        cardPerPrice = state[401+tempVal:406+tempVal]
                        if ((cardPrice > 0).any() or (cardPerPrice > 0).any()) and checkBuyOrientCard(state[300:306], state[306:311], cardPrice, cardPerPrice, state[311]):
                            if state[392+tempVal] == 0 or (state[306:311] > 0).any():
                                validActions[3] = 1
                                break
    
    elif phase[1] == 1: # Lấy nguyên liệu
        takenStocks = state[535:540]
        boardStocks = state[0:6]
        temp = np.where(boardStocks[0:5] > 0)[0] + 4
        validActions[temp] = 1

        s_ = np.sum(takenStocks)
        if s_ == 1:
            t_ = np.where(takenStocks==1)[0][0]
            if boardStocks[t_] < 3: # Không thể lấy double
                validActions[t_+4] = 0

        elif s_ == 2:
            temp_ = np.where(takenStocks == 1)[0] + 4
            validActions[temp_] = 0

        if np.sum(state[300:306]) > 9:
            validActions[9] = 0
    
    elif phase[2] == 1: # Úp thẻ
        # Check thẻ base
        for i in range(3):
            if state[294+i] == 1: # Còn thẻ ẩn
                validActions[10+i] = 1
                tempVal = 4*i
                validActions[13+tempVal:17+tempVal] = 1
            else:
                tempValLv = 4*i
                for j in range(4):
                    tempVal = 44*i + 11*j
                    cardPrice = state[42+tempVal:47+tempVal]
                    if (cardPrice > 0).any():
                        validActions[13+tempValLv+j] = 1
        
        # Check thẻ orient
        for i in range(3):
            if state[297+i] == 1: # Còn thẻ orient ẩn
                validActions[25+i] = 1
                tempVal = 2*i
                validActions[28+tempVal:30+tempVal] = 1
            else:
                tempValLv = 2*i
                for j in range(2):
                    tempVal = 42*i + 21*j
                    cardPrice = state[179+tempVal:184+tempVal]
                    cardPerPrice = state[184+tempVal:189+tempVal]
                    if (cardPrice > 0).any() or (cardPerPrice > 0).any():
                        validActions[28+tempValLv+j] = 1
    
    elif phase[3] == 1: # Mua thẻ
        if (state[547:552] > 0).any():
            validActions[np.where(state[547:552]>0)[0]+4] = 1
        else:
            for i in range(12): # Kiểm tra 12 thẻ trên bàn
                tempVal = 11*i
                cardPrice = state[42+tempVal:47+tempVal]
                if (cardPrice > 0).any() and checkBuyBaseCard(state[300:306], state[306:311], cardPrice, state[311]):
                    validActions[34+i] = 1
            
            for i in range(3): # Kiểm tra 3 thẻ thường úp
                tempVal = 11*i
                cardPrice = state[358+tempVal:363+tempVal]
                if (cardPrice > 0).any() and checkBuyBaseCard(state[300:306], state[306:311], cardPrice, state[311]):
                    validActions[52+i] = 1
            
            for i in range(6): # Kiểm tra 6 thẻ Orient
                tempVal = 21*i
                cardPrice = state[179+tempVal:184+tempVal]
                cardPerPrice = state[184+tempVal:189+tempVal]
                if ((cardPrice > 0).any() or (cardPerPrice > 0).any()) and checkBuyOrientCard(state[300:306], state[306:311], cardPrice, cardPerPrice, state[311]):
                    if state[175+tempVal] == 0 or (state[306:311] > 0).any():
                        validActions[46+i] = 1
            
            for i in range(3): # Kiểm tra 3 thẻ Orient úp
                tempVal = 21*i
                cardPrice = state[396+tempVal:401+tempVal]
                cardPerPrice = state[401+tempVal:406+tempVal]
                if ((cardPrice > 0).any() or (cardPerPrice > 0).any()) and checkBuyOrientCard(state[300:306], state[306:311], cardPrice, cardPerPrice, state[311]):
                    if state[392+tempVal] == 0 or (state[306:311] > 0).any():
                        validActions[52+i] = 1
    
    elif phase[4] == 1: # Trả nguyên liệu
        validActions[np.where(state[300:305]>0)[0]+55] = 1

    elif phase[5] == 1: # Chọn nguyên liệu vv để ghép đôi
        validActions[np.where(state[306:311]>0)[0]+60] = 1
    
    elif phase[6] == 1: # Chọn thẻ để lấy free
        cardLv = np.where(state[545:547]==1)[0][0]

        tempValLv = 4*cardLv
        for j in range(4):
            tempVal = 11*j + 44*cardLv
            cardPrice = state[42+tempVal:47+tempVal]
            if (cardPrice > 0).any():
                validActions[65+tempValLv+j] = 1
        
        tempValLv = 2*cardLv
        for j in range(2):
            tempVal = 21*j + 42*cardLv
            cardPrice = state[179+tempVal:184+tempVal]
            cardPerPrice = state[184+tempVal:189+tempVal]
            if (cardPrice > 0).any() or (cardPerPrice > 0).any():
                validActions[73+tempValLv+j] = 1
    
    elif phase[7] == 1: # Chọn thẻ noble để úp
        for i in range(5):
            tempVal = 6*i
            if (state[7+tempVal:12+tempVal] > 0).any():
                validActions[77+i] = 1
    
    return validActions


@njit
def stepEnv(action, env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3):
    phase = env[106]

    if phase == 0: # Lựa chọn hành động
        if action == 0:
            env[105] += 1
        else:
            env[106] = action
    
    elif phase == 1: # Lấy nguyên liệu
        checkP1 = False
        if action == 9:
            checkP1 = True
        else:
            gem = action - 4

            takenStocks = env[107:112]
            takenStocks[gem] += 1 # Thêm vào nguyên liệu đã lấy trong turn

            pIdx = env[105] % 4
            tempVal = 19*pIdx
            pStocks = env[29+tempVal:35+tempVal]
            pStocks[gem] += 1 # Thêm nguyên liệu cho người chơi hiện tại

            bStocks = env[0:6]
            bStocks[gem] -= 1 # Trừ nguyên liệu ở bàn chơi

            s_ = np.sum(takenStocks)
            if s_ == 1:
                # Còn đúng một loại nguyên liệu và nguyên liệu đó có số lượng < 3
                if bStocks[gem] < 3 and (np.sum(bStocks[0:5]) - bStocks[gem]) == 0:
                    checkP1 = True
            elif s_ == 2:
                # Lấy double hoặc không còn nguyên liệu nào khác 2 cái vừa lấy
                if np.max(takenStocks) == 2 or (np.sum(bStocks[0:5]) - np.sum(bStocks[np.where(takenStocks==1)[0]])) == 0:
                    checkP1 = True
            elif s_ == 3: # Đã lấy 3 nguyên liệu
                checkP1 = True
        
        if checkP1:
            env[107:112] = 0
            if np.sum(pStocks) > 10:
                env[106] = 4
            else:
                env[106] = 0
                env[105] += 1
    
    elif phase == 2: # Úp thẻ
        bStocks = env[0:6]
        pIdx = env[105] % 4
        tempVal = 19*pIdx
        pStocks = env[29+tempVal:35+tempVal]
        tempValHiCa = 42 + tempVal
        posP = np.where(env[tempValHiCa:tempValHiCa+3]==-1)[0][0] + tempValHiCa

        if bStocks[5] > 0: # Trên bàn còn nguyên liệu gold
            pStocks[5] += 1
            bStocks[5] -= 1
        
        if action == 10: # Úp thẻ ẩn cấp 1:
            env[posP] = lv1[lv1[-1]]
            lv1[-1] += 1
        elif action == 11: # Úp thẻ ẩn cấp 2:
            env[posP] = lv2[lv2[-1]]
            lv2[-1] += 1
        elif action == 12: # Úp thẻ ẩn cấp 3:
            env[posP] = lv3[lv3[-1]]
            lv3[-1] += 1
        elif action == 25: # Úp thẻ Orient ẩn cấp 1
            env[posP] = oriLv1[oriLv1[-1]]
            oriLv1[-1] += 1
        elif action == 26: # Úp thẻ Orient ẩn cấp 2
            env[posP] = oriLv2[oriLv2[-1]]
            oriLv2[-1] += 1
        elif action == 27: # Úp thẻ Orient ẩn cấp 3
            env[posP] = oriLv3[oriLv3[-1]]
            oriLv3[-1] += 1
        else:
            if action < 25:
                posE = action - 2
            else:
                posE = action - 5
            
            cardId = env[posE]
            env[posP] = cardId
            fillCard(posE, cardId, env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3)
        
        if np.sum(pStocks) > 10:
            env[106] = 4
        else:
            env[106] = 0
            env[105] += 1
    
    elif phase == 3: # Mua thẻ
        bStocks = env[0:6]
        pIdx = env[105] % 4
        tempVal = 19*pIdx
        pStocks = env[29+tempVal:35+tempVal]
        oriGoldId = 40+tempVal
        pPerStocks = env[35+tempVal:oriGoldId]
        checkBought = False

        if action < 9:
            gem = action - 4
            env[118+gem] -= 1
            env[124] += 1

            pStocks[0:5] -= env[118:123]
            bStocks[0:5] += env[118:123]
            pStocks[5] -= env[123]
            bStocks[5] += env[123]
            env[oriGoldId] -= env[124]
            env[112+pIdx] -= env[124]/2
            env[118:125] = 0

            posE = env[125]
            cardId = env[posE]
            env[125] = -1

            checkBought = True
        
        else:
            # Lấy ra posE và cardId
            if action < 52:
                posE = action - 23
            else:
                posE = -10 + tempVal + action
            
            cardId = env[posE]

            # Lấy ra cardPrice và cardPerPrice
            if cardId < 90: # Thẻ thường
                cardInfor = normalCardInfor[cardId]
                cardPrice = cardInfor[2:7]
                cardPerPrice = np.full(5, 0).astype(np.int32)
            else: # Thẻ orient
                cardInfor = orientCardInfor[cardId-90]
                cardPrice = cardInfor[5:10]
                cardPerPrice = cardInfor[10:15]
            
            # Check xem có cần truyền vào bot hay không
            if (cardPerPrice > 0).any(): # Trừ nguyên liệu vv
                pPerStocks -= cardPerPrice
                env[112+pIdx] -= 2
                checkBought = True
            else: # Mua bằng nl không vv
                check = True
                nlMat = (cardPrice > pPerStocks) * (cardPrice - pPerStocks)
                nlBt = np.minimum(nlMat, pStocks[0:5])
                nlG = np.sum(nlMat - nlBt)

                # Lấy ra a: nl gold thường, b: nl gold thẻ Orient
                if nlG > pStocks[5]: # Cần trả cả nl gold Orient
                    a = pStocks[5]
                    b = nlG - a
                else: # Chỉ cần trả nl gold thường
                    a = nlG
                    b = 0
                
                # Chỉnh lại các thành phần trong giá
                if b % 2 == 1:
                    if a > 0:
                        a -= 1
                        b += 1
                    else:
                        if (nlBt > 0).any():
                            if np.sum(nlBt) == np.max(nlBt): # Thay nl thường = nl Gold orient
                                nlBt[np.where(nlBt==np.max(nlBt))[0][0]] -= 1
                                b += 1
                            else: # Phải trả nhiều hơn 1 loại nl thường, cần truyền cho bot
                                check = False
                        # Nếu ko cần trả nl thường nào
                # Nếu nl gold Orient là số chẵn

                if check: # Trả nl cho thẻ
                    pStocks[0:5] -= nlBt # Trả nguyên liệu
                    bStocks[0:5] += nlBt
                    pStocks[5] -= a
                    bStocks[5] += a
                    env[oriGoldId] -= b # Trừ số nl gold thẻ orient
                    if b % 2 == 1: # Nếu b lẻ thì phải trừ 1 nữa
                        env[oriGoldId] -= 1
                        b += 1
                    
                    env[112+pIdx] -= b/2
                    
                    checkBought = True
                else: # Setup để truyền vào cho bot
                    env[118:123] = nlBt
                    env[123] = a
                    env[124] = b
                    env[125] = posE
        
        if checkBought: # Đã mua thẻ
            if posE < 29: # Mua thẻ trên bàn
                fillCard(posE, cardId, env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3)
            else: # Mua thẻ đang úp
                env[posE] = -1
            
            env[112+pIdx] += 1

            if cardId < 90:
                cardInfor = normalCardInfor[cardId]

                env[41+tempVal] += cardInfor[0] # Cộng điểm
                pPerStocks[cardInfor[1]] += 1 # Tăng nguyên liệu vĩnh viễn lên 1
                checkNoble(env, tempVal, pPerStocks)

                env[105] += 1
                env[106] = 0
            
            else: # Mua thẻ orient
                cardInfor = orientCardInfor[cardId-90]
                env[41+tempVal] += cardInfor[0] # Cộng điểm

                # Check nguyên liệu bonus
                if cardInfor[1] < 5:
                    pPerStocks[cardInfor[1]] += cardInfor[2]
                    checkNoble(env, tempVal, pPerStocks)
                elif cardInfor[1] == 5:
                    env[oriGoldId] += 2
                else: # Flexible
                    env[106] = 5
                
                if cardInfor[3] != 0:
                    if cardInfor[3] == 1 and ((env[11:15] != -1).any() or (env[23:25] != -1).any()):
                        env[117] = 1
                        if env[106] == 3:
                            env[106] = 6
                    elif cardInfor[3] == 2 and ((env[15:19] != -1).any() or (env[25:27] != -1).any()):
                        env[117] = 2
                        if env[106] == 3:
                            env[106] = 6
                
                if cardInfor[4] == 1 and (env[6:11] != -1).any():
                    env[106] = 7
                
                # Check điều kiện next turn
                if env[106] == 3:
                    env[105] += 1
                    env[106] = 0

        # Nếu chưa mua thẻ (đã setup ở trên rồi)
    
    elif phase == 4: # Trả nguyên liệu
        gem = action - 55
        bStocks = env[0:6]
        pIdx = env[105] % 4
        tempVal = 19*pIdx
        pStocks = env[29+tempVal:35+tempVal]

        pStocks[gem] -= 1
        bStocks[gem] += 1
        if np.sum(pStocks) < 11: # Thỏa mãn điều này thì sang turn mới
            env[105] += 1
            env[106] = 0
    
    elif phase == 5: # chọn nguyên liệu để ghép đôi (thẻ orient nguyên liệu flexible)
        gem = action - 60
        pIdx = env[105] % 4
        tempVal = 19*pIdx
        pStocks = env[29+tempVal:35+tempVal]
        pPerStocks = env[35+tempVal:40+tempVal]

        pPerStocks[gem] += 1
        checkNoble(env, tempVal, pPerStocks)

        if env[117] != 0: # Được lấy 1 thẻ free
            env[106] = 6
        else:
            env[106] = 0
            env[105] += 1
    
    elif phase == 6: # chọn thẻ để lấy free
        pIdx = env[105] % 4
        tempVal = 19*pIdx
        pStocks = env[29+tempVal:35+tempVal]
        oriGoldId = 40+tempVal
        pPerStocks = env[35+tempVal:oriGoldId]
        env[117] = 0

        if action < 73:
            posE = action - 54
        else:
            posE = action - 50
        
        cardId = env[posE]
        fillCard(posE, cardId, env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3)

        env[112+pIdx] += 1

        if cardId < 90:
            cardInfor = normalCardInfor[cardId]

            env[41+tempVal] += cardInfor[0] # Cộng điểm
            pPerStocks[cardInfor[1]] += 1 # Tăng nguyên liệu vĩnh viễn lên 1
            checkNoble(env, tempVal, pPerStocks)

            env[105] += 1
            env[106] = 0
        
        else: # Mua thẻ orient
            env[106] = 9999
            cardInfor = orientCardInfor[cardId-90]
            env[41+tempVal] += cardInfor[0] # Cộng điểm

            # Check nguyên liệu bonus
            if cardInfor[1] < 5:
                pPerStocks[cardInfor[1]] += cardInfor[2]
                checkNoble(env, tempVal, pPerStocks)
            elif cardInfor[1] == 5:
                env[oriGoldId] += 2
            else: # Flexible
                env[106] = 5
            
            if cardInfor[3] != 0:
                if cardInfor[3] == 1 and ((env[11:15] != -1).any() or (env[23:25] != -1).any()):
                    env[117] = 1
                    if env[106] == 9999:
                        env[106] = 6
            
            if cardInfor[4] == 1 and (env[6:11] != -1).any():
                env[106] = 7
            
            # Check điều kiện next turn
            if env[106] == 9999:
                env[105] += 1
                env[106] = 0
    
    elif phase == 7:
        pIdx = env[105] % 4
        tempVal = 19*pIdx
        tempValHiCa = 45+tempVal
        posP = np.where(env[tempValHiCa:tempValHiCa+3]==-1)[0][0] + tempValHiCa
        posE = action - 71

        env[posP] = env[posE]
        env[posE] = -1

        env[105] += 1
        env[106] = 0


@njit
def getAgentSize():
    return __AGENT_SIZE__


@njit
def checkEnded(env):
    scoreArr = env[np.array([41, 60, 79, 98])]
    maxScore = np.max(scoreArr)
    if maxScore >= 15 and env[106] == 0 and env[105] % 4 == 0:
        env[116] = 1
        maxScorePlayers = np.where(scoreArr==maxScore)[0]
        if len(maxScorePlayers) == 1:
            return maxScorePlayers[0]
        else:
            playerBoughtCards = env[maxScorePlayers+112]
            min_ = np.min(playerBoughtCards)
            # Trường hợp có nhiều người có cùng điểm, cùng số thẻ thì người đi sau sẽ chiến thắng
            winnerIdx = np.where(playerBoughtCards==min_)[0][-1]
            return maxScorePlayers[winnerIdx]
    else: # Chưa kết thúc game
        return -1


@njit
def getReward(state):
    if state[544] == 0:
        return -1
    else:
        scoreArr = state[np.array([312, 325, 338, 351])]
        maxScore = np.max(scoreArr)
        if maxScore < 15: # Không ai thắng
            return 0
        if scoreArr[0] < maxScore: # Điểm của bản thân không cao nhất
            return 0
        else: # Điểm của bản thân bằng số điểm cao nhất
            maxScorePlayers = np.where(scoreArr==maxScore)[0]
            if len(maxScorePlayers) == 1: # Bản thân là người duy nhất đạt điểm cao nhất
                return 1
            else:
                playerBoughtCards = state[maxScorePlayers+540]
                min_ = np.min(playerBoughtCards)
                if playerBoughtCards[0] > min_: # Số thẻ của bản thân nhiều hơn
                    return 0
                else: # Bản thân mua số lượng thẻ ít nhất
                    lstChk = maxScorePlayers[np.where(playerBoughtCards==min_)[0]]
                    if len(lstChk) == 1: # Bản thân là người duy nhất có số lượng thẻ ít nhất
                        return 1
                    else: # Phải xét vị trí của bản thân
                        selfId = np.where(state[523:527] == 1)[0][0]
                        if selfId + lstChk[1] >= 4: # Chứng tỏ bản thân đi sau cùng trong lst
                            return 1
                        else: # Chứng tỏ có ít nhất một người trong list đi sau bản thân
                            return 0



def one_game(listAgent, perData):
    env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3 = initEnv()
    tempData = []
    for _ in range(__AGENT_SIZE__):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
        tempData.append(dataOnePlayer)

    winner = -1
    while env[105] < 400:
        pIdx = env[105] % 4
        action, tempData[pIdx], perData = listAgent[pIdx](getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
        stepEnv(action, env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3)
        winner = checkEnded(env)
        if winner != -1:
            break
    
    env[116] = 1
    
    for pIdx in range(4):
        env[105] = pIdx
        action, tempData[pIdx], perData = listAgent[pIdx](getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
    
    return winner, perData


@njit
def numba_one_game(p0, p1, p2, p3, perData, pIdOrder):
    env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3 = initEnv()
    tempData = []
    for _ in range(__AGENT_SIZE__):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
        tempData.append(dataOnePlayer)
    
    winner = -1
    while env[105] < 400:
        pIdx = env[105] % 4
        if pIdOrder[pIdx] == 0:
            action, tempData[pIdx], perData = p0(getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 1:
            action, tempData[pIdx], perData = p1(getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 2:
            action, tempData[pIdx], perData = p2(getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 3:
            action, tempData[pIdx], perData = p3(getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
        
        stepEnv(action, env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3)
        winner = checkEnded(env)
        if winner != -1:
            break
    
    env[116] = 1
    
    for pIdx in range(4):
        env[105] = pIdx
        if pIdOrder[pIdx] == 0:
            action, tempData[pIdx], perData = p0(getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 1:
            action, tempData[pIdx], perData = p1(getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 2:
            action, tempData[pIdx], perData = p2(getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 3:
            action, tempData[pIdx], perData = p3(getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3), tempData[pIdx], perData)
    
    return winner, perData



def normal_main(listAgent, times, perData, printMode=False, k=100):
    if len(listAgent) != __AGENT_SIZE__:
        raise Exception('Hệ thống chỉ cho phép có đúng 4 người chơi!!!')
    
    numWin = np.full(5, 0)
    pIdOrder = np.arange(__AGENT_SIZE__)
    for _ in range(times):
        if printMode and _ != 0 and _ % k == 0:
            print(_, numWin)

        np.random.shuffle(pIdOrder)
        shuffledListAgent = [listAgent[i] for i in pIdOrder]
        winner, perData = one_game(shuffledListAgent, perData)

        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    
    if printMode:
        print(_+1, numWin)

    return numWin, perData


@njit
def numba_main(p0, p1, p2, p3, times, perData, printMode=False, k=100):
    numWin = np.full(5, 0)
    pIdOrder = np.arange(__AGENT_SIZE__)
    for _ in range(times):
        if printMode and _ != 0 and _ % k == 0:
            print(_, numWin)

        np.random.shuffle(pIdOrder)
        winner, perData = numba_one_game(p0, p1, p2, p3, perData, pIdOrder)

        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    
    if printMode:
        print(_+1, numWin)

    return numWin, perData



def randomBot(state, tempData, perData):
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    return validActions[idx], tempData, perData


@njit
def numbaRandomBot(state, tempData, perData):
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    return validActions[idx], tempData, perData


@njit
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3 = initEnv()

    winner = -1
    while env[105] < 400:
        pIdx = env[105] % 4
        p_state = getAgentState(env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3)
        list_action = getValidActions(p_state)

        if list_other[pIdx] == -1:
            action, per_player = p0(p_state, per_player)
        elif list_other[pIdx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[pIdx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[pIdx] == 3:
            action, per3 = p3(p_state, per3)

        if list_action[action] != 1:
            raise Exception("Action không hợp lệ")

        stepEnv(action, env, lv1, lv2, lv3, oriLv1, oriLv2, oriLv3)
        winner = checkEnded(env)
        if winner != -1:
            break

    env[116] = 1

    for pIdx in range(4):
        env[105] = pIdx
        if list_other[pIdx] == -1:
            p_state = getAgentState(env, lv1, lv2, lv3)
            action, per_player = p0(p_state, per_player)

    check = False
    if winner != -1 and list_other[winner] == -1:
        check = True

    return check, per_player


@njit
def n_game_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3)
        win += winner

    return win, per_player


import importlib.util, json, sys
from setup import SHOT_PATH


def load_module_player(player):
    return importlib.util.spec_from_file_location('Agent_player',
    f"{SHOT_PATH}Agent/{player}/Agent_player.py").loader.load_module()


@jit
def numba_main_2(p0, n_game, per_player, level):
    list_other = np.array([1, 2, 3, -1])
    if level == 0:
        per_agent_env = np.array([0])
        return n_game_numba(p0, n_game, per_player, list_other, per_agent_env, per_agent_env, per_agent_env, numbaRandomBot, numbaRandomBot, numbaRandomBot)
    else:
        env_name = sys.argv[1]
        dict_level = json.load(open(f'{SHOT_PATH}Log/level_game.json'))
        if str(level) not in dict_level[env_name]:
            raise Exception('Hiện tại không có level này')

        lst_agent_level = dict_level[env_name][str(level)][2]

        p1 = load_module_player(lst_agent_level[0]).Agent
        p2 = load_module_player(lst_agent_level[1]).Agent
        p3 = load_module_player(lst_agent_level[2]).Agent
        per_level = []
        for id in range(getAgentSize()-1):
            data_agent_env = list(np.load(f'{SHOT_PATH}Agent/{lst_agent_level[id]}/Data/{env_name}_{level}/Train.npy',allow_pickle=True))
        per_level.append(data_agent_env)

        return n_game_numba(p0, n_game, per_player, list_other, per_level[0], per_level[1], per_level[2], p1, p2, p3)
