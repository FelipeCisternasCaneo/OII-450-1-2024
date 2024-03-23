import numpy as np
import math

class QLearning():
    
    def __init__(self, iterMax, paramsML, epsilon = 0.005, qlAlpha = 0.1):
        
        self.__actions = paramsML['DS_actions']
        self.__gamma = paramsML['gamma']
        self.__qlAlphaType = paramsML['qlAlphaType']
        self.__rewardType = paramsML['rewardType']
        self.__minmax = paramsML['MinMax']
        self.__policy = paramsML['policy']
        stateQ = paramsML['stateQ']
        self.__iterMax = iterMax
        self.__epsilon = epsilon
        self.__qlAlpha = qlAlpha
        self.__bestMetric = 999999999
        self.__W = 10
        self.__Qvalues = np.zeros( shape=(stateQ , len(self.__actions)) )
        self.__visitas = np.zeros( shape=(stateQ , len(self.__actions)) )
        
    def setActions(self, actions):
        self.__actions = actions
    def getActions(self):
        return self.__actions
    def setGamma(self, gamma):
        self.__gamma = gamma
    def getGamma(self):
        return self.__gamma
    def setQlAlphaType(self, qlAlphaType):
        self.__qlAlphaType = qlAlphaType
    def getQlAlphaType(self):
        return self.__qlAlphaType
    def setRewardType(self, rewardType):
        self.__rewardType = rewardType
    def getRewardType(self):
        return self.__rewardType
    def setItermax(self, itermax):
        self.__iterMax = itermax
    def getItermax(self):
        return self.__iterMax
    def setEpsilon(self, epsilon):
        self.__epsilon = epsilon
    def getEpsilon(self):
        return self.__epsilon
    def setQlAlpha(self, qlAlpha):
        self.__qlAlpha = qlAlpha
    def getQlAlpha(self):
        return self.__qlAlpha
    def setBestmetric(self, bestMetric):
        self.__bestMetric = bestMetric
    def getBestmetric(self):
        return self.__bestMetric
    def setW(self, w):
        self.__W = w
    def getW(self):
        return self.__W
    def setMinmax(self, minmax):
        self.__minmax = minmax
    def getMinmax(self):
        return self.__minmax
    def setPolicy(self, policy):
        self.__policy = policy
    def getPolicy(self):
        return self.__policy
    def setQvalues(self, qvalues):
        self.__Qvalues = qvalues
    def getQvalues(self):
        return self.__Qvalues
    def setVisitas(self, visitas):
        self.__visitas = visitas
    def getVisitas(self):
        return self.__visitas
    
    def getReward(self, metric):
        
        if self.getMinmax() == "min":
            
            if self.getRewardType() == "withPenalty1":
                if self.getBestmetric() > metric:
                    self.setBestmetric(metric)
                    return 1
                return -1
            
            elif self.getRewardType() == "withoutPenalty1":
                if self.getBestmetric() > metric:
                    self.setBestmetric(metric)
                    return 1
                return 0
            
            elif self.getRewardType() == "globalBest":
                if self.getBestmetric() > metric:
                    self.setBestmetric(metric)
                    return self.getW() / self.getBestmetric()
                return 0
            
            elif self.getRewardType() == "rootAdaptation":
                if self.getBestmetric() > metric:
                    self.setBestmetric(metric)
                    return math.sqrt(metric)
                return 0
            
            elif self.getRewardType() == "scalatingMultiplicativeAdaptation":
                if self.getBestmetric() > metric:
                    self.setBestmetric(metric)
                    return self.getW() * self.getBestmetric()
                return 0
            
            
        elif self.getMinmax() == "max":
            
            if self.getRewardType() == "withPenalty1":
                if self.getBestmetric() < metric:
                    self.setBestmetric(metric)
                    return 1
                return -1
            
            elif self.getRewardType() == "withoutPenalty1":
                if self.getBestmetric() < metric:
                    self.setBestmetric(metric)
                    return 1
                return 0
            
            elif self.getRewardType() == "globalBest":
                if self.getBestmetric() < metric:
                    self.setBestmetric(metric)
                    return self.getW() / self.getBestmetric()
                return 0
            
            elif self.getRewardType() == "rootAdaptation":
                if self.getBestmetric() < metric:
                    self.setBestmetric(metric)
                    return math.sqrt(metric)
                return 0
            
            elif self.getRewardType() == "scalatingMultiplicativeAdaptation":
                if self.getBestmetric() < metric:
                    self.setBestmetric(metric)
                    return self.getW() * self.getBestmetric()
                return 0
            
                
        
    def getAction(self, state):
        
        # e-greedy
        if self.getPolicy() == "e-greedy":
            prob = np.random.uniform(low = 0.0, high = 1.0)     
            
            if prob <= self.getEpsilon():
                actionRandom = np.random.randint(low = 0, high = self.getQvalues().shape[0])
                return actionRandom
            else:
                maximo = np.amax(self.getQvalues()[state])
                indices = np.where(self.getQvalues()[state,:] == maximo)[0]
                
                return np.random.choice(indices)
        
        elif self.getPolicy() == "greedy":
            return np.argmax(self.getQvalues()[state])
        
        elif self.getPolicy() == "e-soft":
            prob = np.random.uniform(low = 0.0, high = 1.0)
            if prob > self.getEpsilon(): 
                return np.random.randint(low = 0, high = self.getQvalues().shape[0])
            else:
                maximo = np.amax(self.getQvalues(), axis=1)
                indices = np.where(self.getQvalues()[state,:] == maximo[state])[0]
                return np.random.choice(indices)
            
        # elif self.getPolicy() == "softMax-rulette-elitists":
        #     ordenInvertido = np.multiply(self.getQvalues()[state], -1)
        #     sort = np.argsort(ordenInvertido)
        #     cant_mejores = int(sort.shape[0] * 0.25)
        #     rulette_elisits = sort[0 : cant_mejores]
        #     return np.random.choice(rulette_elisits)
    
    def actualizarVisitas(self, action, state):
        visitas = self.getVisitas()
        visitas[state,action] +=1
        self.setVisitas(visitas)
        
    def getAlpha(self, state, action, iter):
        
        if self.getQlAlphaType() == "static":
            return self.getQlAlpha()
        
        elif self.getQlAlphaType() == "iteration":
            return 1 - ( 0.9 * ( iter / self.getItermax() ) )
        
        elif self.getQlAlphaType() == "visits":
            return ( 1 / ( 1 + self.getVisitas()[state,action] ) )
        
    def updateQtable(self, metric, action, state, iter):
        
        qTable = self.getQvalues()
        
        reward = self.getReward(metric)
        
        alpha = self.getAlpha(state, action, iter)
        
        qTable[state,action] = self.getQvalues()[state,action] + alpha * ( reward + ( self.getGamma() * max(self.getQvalues()[state]) ) - self.getQvalues()[state,action]  )
        
        self.actualizarVisitas(action, state)
        
        self.setQvalues(qTable)