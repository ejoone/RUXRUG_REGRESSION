#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18 April 2022

@original author: sibirbil
@author: Junhee Lee 496160
"""
import copy
import time
import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeRegressor
from auxClasses import Clause, Rule

class RUXClassifier:
    
        def __init__(self, rf=None, 
                     ada=None, 
                     eps=1.0e-2,
                     threshold=1.0e-6, #weight threshold
                     use_ada_weights=False,
                     rule_length_cost=False, 
                     false_negative_cost=False,
                     negative_label=1.0, # For identifying false negatives
                     solver='glpk',
                     random_state=2516):
            
            self.eps = eps
            self.threshold = threshold
            self.wscale = 1.0 #weight scale
            self.vscale = 1.0 #auxilary variable scale
            self.fittedDTs = {} #dictionary (key: treeno / value: fitTree)
            self.solver = solver
            self.randomState = random_state
            self.initNumOfRules = 0
            self.rules = {}
            self.ruleInfo = {} #"location" of the rule inside the RF
            self.K = None # number of classes
            self.labelToInteger = {} # mapping classes to integers
            self.integerToLabel= {} # mapping integers to classes
            self.vecY = None # vector of R(x_i)
            self.majorityClass = None # which class is the majority
            self.missedXvals = None        
            self.numOfMissed = None
            # The following three vectors keep the Ahat and A matrices
            # Used for CSR sparse matrices
            self.yvals = np.empty(shape=(0), dtype=np.float) #label of y (1 or -1 for binary)
            self.rows = np.empty(shape=(0), dtype=np.int32)
            self.cols = np.empty(shape=(0), dtype=np.int32) 
            # The cost of each rule is stored
            self.costs = np.empty(shape=(0), dtype=np.float)
            self.ruleLengthCost = rule_length_cost
            self.falseNegativeCost = false_negative_cost
            self.negativeLabel = negative_label
            self.useADAWeights = use_ada_weights
            self.estimatorWeights = [] # Used with AdaBoost
            # For time keeping        
            self.fitTime = 0
            self.predictTime = 0
            # Classifier type
            self.classifier = ''
            
            self._checkOptions(rf, ada)

            self. vecR = np.empty(shape=(0), dtype=np.float)
            self. yData = None


        def _checkOptions(self, rf, ada):
            """

            :param rf: fitted random forest instance (trained)
            :param ada: fitted ada instance (trained)
            :return: None
            """
            
            if (rf == None and ada == None):
                print('RF or ADA should be provided')
                print('Exiting...')
                return None
            
            if (rf != None and ada != None):
                print('Both RF and ADA are provided')
                print('Proceeding with RF')
                ada = None
                
            if (rf != None):
                self.classifier = 'RF'
                
                if (self.useADAWeights):
                    print('Estimator weights work only with ADA')
                    self.useADAWeights = False
                    
                if (np.sum([self.ruleLengthCost, self.falseNegativeCost]) > 1):
                    print('Works with only one type of cost')
                    print('Proceeding with rule length')
                    self.falseNegativeCost = False
                    self.ruleLengthCost = True

                for treeno, fitTree in enumerate(rf.estimators_):
                    self.initNumOfRules += fitTree.get_n_leaves()
                    self.fittedDTs[treeno] = fitTree
                
            if (ada != None):
                self.classifier = 'ADA'
                
                if (np.sum([self.ruleLengthCost, 
                            self.falseNegativeCost, 
                            self.useADAWeights]) > 1):
                    print('Works with only one type of cost')
                    print('Proceeding with estimator weights')
                    self.falseNegativeCost = False
                    self.ruleLengthCost = False
                    self.useADAWeights = True

                if (self.useADAWeights):
                    self.estimatorWeights = (1.0/(ada.estimator_weights_+1.0e-4))
                    
                for treeno, fitTree in enumerate(ada.estimators_):
                    self.initNumOfRules += fitTree.get_n_leaves()
                    self.fittedDTs[treeno] = fitTree

        def _cleanup(self):
            
            self.fittedDTs = {}   
            self.rules = {}      
            self.ruleInfo = {}
            self.labelToInteger = {} 
            self.integerToLabel= {}
            self.missedXvals = None        
            self.numOfMissed = None
            self.yvals = np.empty(shape=(0), dtype=np.float)
            self.rows = np.empty(shape=(0), dtype=np.int32)
            self.cols = np.empty(shape=(0), dtype=np.int32) 
            self.costs = np.empty(shape=(0), dtype=np.float)
            self.estimatorWeights = []            
                    
        def _getRule(self, fitTree, nodeid):
            """

            :param fitTree: individual tree estimator in the RF/ADA instance
            :param nodeid: primary key for each node in the fitTree instance
            :return: information of rule (division of node clause + depth)
            """
            
            if (fitTree.tree_.feature[0] == -2): # No rule case
                return Rule()
            left = fitTree.tree_.children_left
            right = fitTree.tree_.children_right
            threshold = fitTree.tree_.threshold
        
            def recurse(left, right, child, returnRule=None):
                if returnRule is None:
                    returnRule = Rule()                
                if child in left: # 'l'
                    parent = np.where(left == child)[0].item()
                    clause = Clause(feature=fitTree.tree_.feature[parent], 
                                    ub=threshold[parent])
                else: # 'r'               
                    parent = np.where(right == child)[0].item()
                    clause = Clause(feature=fitTree.tree_.feature[parent], 
                                    lb=threshold[parent])                
                returnRule.addClause(clause)
                if parent == 0:
                    return returnRule
                else:
                    return recurse(left, right, parent, returnRule)
        
            retRule = recurse(left, right, nodeid)
        
            return retRule
        
        # y = vector of training data y
        def _getMatrix(self, X, y, fitTree, treeno):
            """
            :param X: training data X (attributes / explanatory variables)
            :param y: training outcome variable Y
            :param fitTree: tree classifier instance inside the RF or ADA
            :param treeno: corresponding id of the fitTree
            :return:
            """
            #change start (keep codes related to cols & rows, but replace codes for yvals and Ahat to the codes for (our) R_j)
            if (len(self.cols) == 0):
                col = 0
            else:
                col = max(self.cols) + 1 # Next column

            y_rules = fitTree.apply(X) # Tells us which sample is in which leaf

            for leafno in np.unique(y_rules):
                covers = np.where(y_rules == leafno)[0]
                leafYvals = y[covers] # y values of the samples in the leaf
                leafYval = np.average(leafYvals)
                self.vecR = np.append(self.vecR, leafYval) #vecR = vector of R_j for all j in J.

                self.rows = np.hstack((self.rows, covers))
                self.cols = np.hstack((self.cols, np.ones(len(covers), dtype=np.int32)*col))

                if (self.falseNegativeCost):
                    cost = 1.0

                elif (self.ruleLengthCost):
                    tempRule = self._getRule(fitTree, leafno)
                    cost = tempRule.length()
                elif (self.useADAWeights):
                    cost = self.estimatorWeights[treeno]
                else:
                    cost = 1.0
                self.costs = np.append(self.costs, cost)
                self.ruleInfo[col] = (treeno, leafno, leafYval)
                col += 1

        def _getMatrices(self, X, y):
            """

            :param X: training data X
            :param y: training data y
            :return: iteration of _getMatrix method using all trees generated by tree-ensemble methods (RF or ADA)
            """
            for treeno, fitTree in enumerate(self.fittedDTs.values()):                    
                self._getMatrix(X, y, fitTree, treeno)

        def _preprocess(self, X, y):
            """
            :param X: training explanatory variables
            :param y: training outcome variable
            :return: define vecY (=R_j)
            """

            classes, classCounts = np.unique(y, return_counts=True)
            self.majorityClass = classes[np.argmax(classCounts)]
            for i, c in enumerate(classes):
                self.labelToInteger[c] = i
                self.integerToLabel[i] = c
            self.K = len(classes)
            n = len(y)
            self.vscale = 1.0
            self.vecY = np.ones((self.K, n))*(-1/(self.K-1))

            for i, c in enumerate(y):
                self.vecY[self.labelToInteger[c], i] = 1

            
        def _fillRules(self, weights, y): #Keep
            """

            :param weights: optimal weights computed by LP
            :return: insert the info about the rule to the self.rules array
            """

            selectedColumns = np.where(weights > self.threshold)[0] # Selected columns
            weightOrder = np.argsort(-weights[selectedColumns]) # Ordered weights
            orderedColumns = selectedColumns[weightOrder] # Ordered indices in terms of weight (rule with largest weight first)

            for i, col in enumerate(orderedColumns):
                treeno, leafno, label = self.ruleInfo[col]
                fitTree = self.fittedDTs[treeno]
                if (fitTree.get_n_leaves()==1):
                    self.rules[i] = Rule(label=label,
                                         clauses=[],
                                         weight=weights[col]) # No rule
                else:
                    self.rules[i] = self._getRule(fitTree, leafno)
                    self.rules[i].label = label
                    self.rules[i].weight = weights[col]                
                    self.rules[i]._cleanRule()

        def _solvePrimal(self, y):
            """

            :param y: training data y
            :return: choose which LP solver to use
            """
            
            if(self.solver == 'glpk'):
                return self._solvePrimalGLPK(y)
            elif (self.solver == 'gurobi'):
                return self._solvePrimalGurobi(y)
            else:
                print('This solver does not exist')
                
                     
        def _solvePrimalGLPK(self, y):
            #GLPK LP solver
            
            data = np.ones(len(self.rows), dtype=np.int32)
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32)


            n, m = max(self.rows)+1, max(self.cols)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale
            # Variables
            vs = cp.Variable(n, nonneg=True)
            ws = cp.Variable(m, nonneg=True)
            AuxR = np.transpose([self.vecR] * n).T
            auxA = A.toarray()
            primal = cp.Problem(cp.Minimize((np.ones(n) * self.vscale) @ vs +
                                            self.costs @ ws),
                                [vs >= y-np.multiply(auxA, AuxR)@ws, vs >= np.multiply(auxA, AuxR)@ws-y])
            primal.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
            
            return ws.value
            
        def _solvePrimalGurobi(self, y):
            #Gurobi LP solver
            #When we obtain the results, we use Gurobi solver.

            data = np.ones(len(self.rows), dtype=np.int32)        
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32)
            # define y_hat variable using A matrix and R_j array (vector)
            
            n, m = max(self.rows)+1, max(self.cols)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale
            # Primal Model
            modprimal = gp.Model('RUG Primal')
            modprimal.setParam('OutputFlag', False)
            # variables
            vs = modprimal.addMVar(shape=int(n), name='vs')
            ws = modprimal.addMVar(shape=int(m), name='ws')
            AuxR = np.transpose([self.vecR] * n).T
            auxA = A.toarray()
            # objective
            modprimal.setObjective((np.ones(n) * self.vscale) @ vs +
                                            self.costs @ ws, GRB.MINIMIZE)
            # constraints
            modprimal.addConstr(vs >= y-np.multiply(auxA, AuxR)@ws, name='abs val Constraints 1')
            modprimal.addConstr(vs >= np.multiply(auxA, AuxR)@ws-y, name='abs val constraints 2')
            modprimal.optimize()
            
            return ws.X
        
        def printRules(self, indices=[]):
            """

            :param indices: list of primary key of the rule
            :return: print detailed information including clauses about the rule corresponding to the indices.
            """
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING! printRules() ###\n')
                print('Do not have that many rules')                
                return
            
            for indx in indices:
                rule = self.rules[indx]
                print('RULE %d:' % (indx))
                if (rule == 'NR'):
                    print('==> No Rule: Set Majority Class')
                else:
                    rule.printRule()
                print('Rule value (R_j): %.4f' % rule.label)
                print('Scaled rule weight: %.4f\n' % rule.weight)
    
        def printWeights(self, indices=[]):
    
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: printWeights() ###\n')
                print('Do not have that many rules')                
                return
            
            for indx in indices:
                rule = self.rules[indx]
                print('RULE %d:' % (indx))
                print('Class: %.0f' % rule.label)
                print('Scaled rule weight: %.4f\n' % rule.weight)
                
        def getWeights(self, indices=[]):
    
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: getWeights() ###\n')
                print('Do not have that many rules')                
                return None 
            
            return [self.rules[indx].weight for indx in indices]    
                
        def predict(self, X, indices=[]):
            """

            :param X: test data X (30% of the total data)
            :param indices: list of primary key of the rule
            :return: predicted y values of test data X (y_hat)
            """

            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: predict() ###\n')
                print('Do not have that many rules')
                return None

            self.missedXvals = []
            self.numOfMissed = 0

            startTime = time.time()
            # TODO: Can be done in parallel
            returnPrediction = []
            for x0 in X:
                sumSampleWeights = 0
                predictedVal=0
                for indx in indices:
                    rule = self.rules[indx]
                    if (rule != 'NR'):
                        if(rule.checkRule(x0)):
                            predictedVal += rule.weight * rule.label # not vecR but one element of vecR correponding to the single rule
                            sumSampleWeights += rule.weight
                if (sumSampleWeights == 0):
                    # Unclassified test sample
                    self.numOfMissed += 1
                    self.missedXvals.append(x0)
                    returnPrediction.append(np.average(self.vecR)) # assigned to an average of vecR
                else:
                    returnPrediction.append(predictedVal)

            endTime = time.time()
            self.predictTime = endTime - startTime

            return returnPrediction
    
        def getAvgRuleLength(self):
            
            return np.mean([rule.length() for rule in self.rules.values()])
            
        def getNumOfRules(self):
            
            return len(self.rules)
    
        def getInitNumOfRules(self):
            
            return self.initNumOfRules
    
        def getNumOfMissed(self):
            
            return self.numOfMissed
    
        def getFitTime(self):
            
            return self.fitTime
    
        def getPredictTime(self):
            
            return self.predictTime
        
        def fit(self, X, y):
            
            if (len(self.cols) != 0):
                self._cleanup()
       
            startTime = time.time()
            
            self._preprocess(X, y)
            self._getMatrices(X, y)
            
            ws = self._solvePrimal(y)

            self._fillRules(ws, y)
            
            endTime = time.time()
            
            self.fitTime = endTime - startTime

            self.yData = y

class RUGClassifier:       
        def __init__(self,
                     eps=1.0e-2,
                     threshold=1.0e-6,
                     max_depth=2,
                     max_RMP_calls=30,
                     rule_length_cost=False,
                     false_negative_cost=False,
                     negative_label=1.0, # For identifying false negatives
                     solver='glpk',
                     random_state=2516):
           
            self.eps = eps
            self.threshold = threshold
            self.wscale = 1.0
            self.vscale = 1.0
            self.fittedDTs = {}
            self.solver = solver
            self.randomState = random_state
            self.rules = {}
            self.ruleInfo = {}
            self.K = None # number of classes
            self.labelToInteger = {} # mapping classes to integers
            self.integerToLabel= {} # mapping integers to classes
            self.vecY = None
            self.majorityClass = None # which class is the majority
            self.missedXvals = None
            self.numOfMissed = None
            self.maxDepth = max_depth
            self.maxRMPcalls = max_RMP_calls
            # The following three vectors keep the Abar and A matrices
            # Used for CSR sparse matrices
            self.yvals = np.empty(shape=(0), dtype=np.float)
            self.rows = np.empty(shape=(0), dtype=np.int32)
            self.cols = np.empty(shape=(0), dtype=np.int32)
            # The cost of each rule is stored
            self.costs = np.empty(shape=(0), dtype=np.float)
            self.ruleLengthCost = rule_length_cost
            self.falseNegativeCost = false_negative_cost
            self.negativeLabel = negative_label
            # For time keeping        
            self.fitTime = 0
            self.predictTime = 0
            
            self._checkOptions()

            self.vecR = np.empty(shape=(0), dtype=np.float)

        def _checkOptions(self):
            
            if (np.sum([self.ruleLengthCost, self.falseNegativeCost]) > 1):
                print('Works with only one type of cost')
                print('Proceeding with rule length')
                self.falseNegativeCost = False
                self.ruleLengthCost = True

        def _cleanup(self):
            
            self.fittedDTs = {}   
            self.rules = {}      
            self.ruleInfo = {}
            self.labelToInteger = {} 
            self.integerToLabel= {}
            self.missedXvals = None        
            self.numOfMissed = None
            self.yvals = np.empty(shape=(0), dtype=np.float)
            self.rows = np.empty(shape=(0), dtype=np.int32)
            self.cols = np.empty(shape=(0), dtype=np.int32) 
            self.costs = np.empty(shape=(0), dtype=np.float)
                    
        def _getRule(self, fitTree, nodeid):
            """

            :param fitTree: individual tree estimator in the RF/ADA instance
            :param nodeid: primary key for each node in the fitTree instance
            :return: information of rule (division of node clause + depth)
            """
            if (fitTree.tree_.feature[0] == -2): # No rule case
                return Rule()
            left = fitTree.tree_.children_left
            right = fitTree.tree_.children_right
            threshold = fitTree.tree_.threshold
        
            def recurse(left, right, child, returnRule=None):
                if returnRule is None:
                    returnRule = Rule()                
                if child in left: # 'l'
                    parent = np.where(left == child)[0].item()
                    clause = Clause(feature=fitTree.tree_.feature[parent], 
                                    ub=threshold[parent])
                else: # 'r'
                    parent = np.where(right == child)[0].item()
                    clause = Clause(feature=fitTree.tree_.feature[parent], 
                                    lb=threshold[parent])                
                returnRule.addClause(clause)
                if parent == 0:
                    return returnRule
                else:
                    return recurse(left, right, parent, returnRule)
        
            retRule = recurse(left, right, nodeid)
        
            return retRule


        def _getInitMatrix(self, X, y, fitTree, treeno):
            """

            :param X: training data X
            :param y: training data y
            :param fitTree: single tree instance
            :param treeno: label of the fittree instances
            :return:
            """

            if (len(self.cols) == 0):
                col = 0
            else:
                col = max(self.cols) + 1  # Next column

            y_rules = fitTree.apply(X)  # Tells us which sample is in which leaf

            for leafno in np.unique(y_rules):
                covers = np.where(y_rules == leafno)[0]

                leafYvals = y[covers]  # y values of the samples in the leaf

                leafYval = np.average(leafYvals)
                self.vecR = np.append(self.vecR, leafYval)

                self.rows = np.hstack((self.rows, covers))
                self.cols = np.hstack((self.cols, np.ones(len(covers), dtype=np.int32) * col))


                if (self.ruleLengthCost):
                    tempRule = self._getRule(fitTree, leafno)
                    cost = tempRule.length()
                elif (self.useADAWeights):
                    cost = self.estimatorWeights[treeno]
                else:
                    cost = 1.0
                self.costs = np.append(self.costs, cost)
                self.ruleInfo[col] = (treeno, leafno, leafYval)
                col += 1

            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale


        def _PSPDT(self, X, y, fitTree, treeno, betas, gammas):

            n, col = max(self.rows)+1, max(self.cols)+1
            y_rules = fitTree.apply(X) # Tells us which sample is in which leaf
            noImprovement = True
            tempvecR = np.empty(shape=(0), dtype=np.float)
            for leafno in np.unique(y_rules):
                covers = np.where(y_rules == leafno)[0]
                # prepare to check the reduced cost
                aij = np.zeros(n)
                leafYvals = y[covers] # y values of the samples in the leaf
                leafYval = np.average(leafYvals)
                tempvecR=np.append(tempvecR, leafYval)
                aij[covers] = 1
                if (self.falseNegativeCost):
                    cost = 1.0
                elif (self.ruleLengthCost):
                    tempRule = self._getRule(fitTree, leafno)
                    cost = tempRule.length()
                else:
                    cost = 1.0
                cost *= self.wscale
                red_cost = np.dot(aij, betas)*leafYval - \
                    np.dot(aij, gammas)*leafYval - cost
                if (red_cost > 0): # Only columns with positive reduced costs are added  
                    self.rows = np.hstack((self.rows, covers))
                    self.cols = np.hstack((self.cols, np.ones(len(covers), dtype=np.int32)*col))
                    self.costs = np.append(self.costs, cost)
                    self.ruleInfo[col] = (treeno, leafno, leafYval)
                    col += 1
                    self.vecR = np.append(self.vecR, leafYval)
                    noImprovement = False
                    
            return noImprovement
             

        def _preprocess(self, X, y):
            classes, classCounts = np.unique(y, return_counts=True)
            self.majorityClass = classes[np.argmax(classCounts)]
            for i, c in enumerate(classes):
                self.labelToInteger[c] = i
                self.integerToLabel[i] = c
            self.K = len(classes)
            n = len(y)
            self.vscale = 1.0
            self.vecY = np.ones((self.K, n))*(-1/(self.K-1))
            for i, c in enumerate(y):
                self.vecY[self.labelToInteger[c], i] = 1        
            
        def _fillRules(self, weights):
            self.rules.clear()
            # print(weights)
            selectedColumns = np.where(weights > self.threshold)[0] # Selected columns
            weightOrder = np.argsort(-weights[selectedColumns]) # Ordered weights
            orderedColumns = selectedColumns[weightOrder] # Ordered indices

            for i, col in enumerate(orderedColumns):
                treeno, leafno, label = self.ruleInfo[col]
                fitTree = self.fittedDTs[treeno]
                if (fitTree.get_n_leaves()==1):
                    self.rules[i] = Rule(label=label,
                                         clauses=[],
                                         weight=weights[col]) # No rule
                else:
                    self.rules[i] = self._getRule(fitTree, leafno)
                    self.rules[i].label = label
                    self.rules[i].weight = weights[col]                
                    self.rules[i]._cleanRule()

        def _solvePrimal(self, y, ws0=[], vs0=[]):

            if (self.solver == 'glpk'):
                return self._solvePrimalGLPK(y,ws0=ws0, vs0=vs0)
            elif (self.solver == 'gurobi'):
                return self._solvePrimalGurobi(y,ws0=ws0, vs0=vs0)
            else:
                print('This solver does not exist')

        def _solvePrimalGLPK(self, y,ws0=[] ,vs0=[]):
            data = np.ones(len(self.rows), dtype=np.int32)
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32)

            n, m = max(self.rows)+1, max(self.cols)+1
            self.wscale = 1.0/np.max(self.costs)
            self.costs *= self.wscale
            # Variables
            vs = cp.Variable(n, nonneg=True)
            ws = cp.Variable(m, nonneg=True)
            if (len(vs0) > 0):
                vs.value = vs0
            if (len(ws0) > 0):
                ws.value = np.zeros(m)
                ws.value[:len(ws0)] = ws0
            AuxR = np.transpose([self.vecR] * n).T
            auxA = A.toarray()
            # Primal Model
            primal = cp.Problem(cp.Minimize((np.ones(n) * self.vscale) @ vs +
                                            self.costs @ ws),
                                [vs + np.multiply(auxA, AuxR) @ ws >= y, vs - np.multiply(auxA, AuxR) @ ws >= -y])
            primal.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})

            betas = primal.constraints[0].dual_value
            gammas = primal.constraints[1].dual_value
            return ws.value, vs.value, betas, gammas

        def _solvePrimalGurobi(self, y, ws0=[], vs0=[]):
            data = np.ones(len(self.rows), dtype=np.int32)
            A = csr_matrix((data, (self.rows, self.cols)), dtype=np.int32)

            n, m = max(self.rows) + 1, max(self.cols) + 1
            self.wscale = 1.0 / np.max(self.costs)
            self.costs *= self.wscale
            # Primal Model
            modprimal = gp.Model('RUG Primal')
            modprimal.setParam('OutputFlag', False)
            # variables
            vs = modprimal.addMVar(shape=int(n), name='vs')
            ws = modprimal.addMVar(shape=int(m), name='ws')
            AuxR = np.transpose([self.vecR] * n).T
            auxA = A.toarray()
            if (len(vs0) > 0):
                vs.setAttr('Start', vs0)
                modprimal.update()
            if (len(ws0) > 0):
                tempws = np.zeros(m)
                tempws[:len(ws0)] = ws0
                ws.setAttr('Start', tempws)
                modprimal.update()
            # objective
            modprimal.setObjective((np.ones(n) * self.vscale) @ vs +
                                            self.costs @ ws, GRB.MINIMIZE)
            # constraints
            modprimal.addConstr(vs + np.multiply(auxA, AuxR) @ ws >= y,
                                name='Ahat Constraints')
            modprimal.addConstr(vs - np.multiply(auxA, AuxR) @ ws >= -y, name='A Constraints')
            modprimal.optimize()
            betas = np.array(modprimal.getAttr(GRB.Attr.Pi)[:n])
            gammas = np.array(modprimal.getAttr(GRB.Attr.Pi)[n:n + n])

            return ws.X, vs.X, betas, gammas


        def printRules(self, indices=[]):
            
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING! printRules() ###\n')
                print('Do not have that many rules')                
                return
            
            for indx in indices:
                rule = self.rules[indx]
                print('RULE %d:' % (indx))
                if (rule == 'NR'):
                    print('==> No Rule: Set Majority Class')
                else:
                    rule.printRule()
                print('Rule value (R_j): %.4f' % rule.label)
                print('Scaled rule weight: %.4f\n' % rule.weight)
    
        def printWeights(self, indices=[]):
    
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: printWeights() ###\n')
                print('Do not have that many rules')                
                return
            
            for indx in indices:
                rule = self.rules[indx]
                print('RULE %d:' % (indx))
                print('Class: %.0f' % rule.label)
                print('Scaled rule weight: %.4f\n' % rule.weight)
                
        def getWeights(self, indices=[]):
    
            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: getWeights() ###\n')
                print('Do not have that many rules')                
                return None 
            
            return [self.rules[indx].weight for indx in indices]    
                
        def predict(self, X, indices=[]):
            #this method computes y_hat using the test sample X and the trained RUG algorithm.

            if (len(indices) == 0):
                indices = self.rules.keys()
            elif (np.max(indices) > len(self.rules)):
                print('\n### WARNING!: predict() ###\n')
                print('Do not have that many rules')
                return None

            self.missedXvals = []
            self.numOfMissed = 0

            startTime = time.time()
            # TODO: Can be done in parallel
            returnPrediction = []
            for x0 in X:
                sumClassWeights = np.zeros(self.K)
                sumSampleWeights = 0
                predictedVal=0

                x0rules = np.empty(shape=(0), dtype=np.float)
                x0rules_weights=np.empty(shape=(0), dtype=np.float)

                for indx in indices:
                    rule = self.rules[indx]
                    x0rules_weights=np.append(x0rules_weights,rule.weight)

                for indx in indices:
                    rule = self.rules[indx]
                    if (rule != 'NR'):
                        if(rule.checkRule(x0)):
                            predictedVal += rule.weight * rule.label
                            sumSampleWeights += rule.weight
                if (sumSampleWeights == 0):
                    # Unclassified test sample
                    self.numOfMissed += 1
                    self.missedXvals.append(x0)
                    returnPrediction.append(np.average(self.vecR)) # assigned to an average of vecR
                else:
                    returnPrediction.append(predictedVal)

            endTime = time.time()
            self.predictTime = endTime - startTime

            return returnPrediction
    
        def getAvgRuleLength(self):
            
            return np.mean([rule.length() for rule in self.rules.values()])
            
        def getNumOfRules(self):
            
            return len(self.rules)
    
        def getNumOfMissed(self):
            
            return self.numOfMissed
    
        def getFitTime(self):
            
            return self.fitTime
    
        def getPredictTime(self):
            
            return self.predictTime
        
        def fit(self, X, y):
            #The commented block of code is for RUG1
            # if (len(self.cols) != 0):
            #     self._cleanup()
            #
            # startTime = time.time()
            #
            # treeno = 0
            # DT = DecisionTreeRegressor(max_depth=self.maxDepth,
            #                            random_state=self.randomState)
            #
            # fitTree = DT.fit(X, y)
            # self.fittedDTs[treeno] = copy.deepcopy(fitTree)
            # self._preprocess(X, y)
            # self._getInitMatrix(X, y, fitTree, treeno)
            # ws, vs, betas, gammas = self._solvePrimal(y=y)
            #
            # self._fillRules(ws)
            # yhat = self.predict(X)
            #
            # # Column generation
            # newDual = betas - gammas
            # ones = np.ones(len(newDual))
            # newDual = (newDual + ones) / 2
            # # new CG weights
            # for cg in range(self.maxRMPcalls):
            #     treeno += 1
            #     fitTree = DT.fit(X, y, sample_weight=newDual)
            #     self.fittedDTs[treeno] = copy.deepcopy(fitTree)
            #     noImprovement = self._PSPDT(X, y, fitTree, treeno, betas, gammas)
            #     if (noImprovement):
            #         break
            #     ws, vs, betas, gammas = self._solvePrimal(y, ws0=ws, vs0=vs)
            #     newDual = betas - gammas
            #     ones = np.ones(len(newDual))
            #     newDual = (newDual + ones) / 2
            #     self._fillRules(ws)
            #     yhat = self.predict(X)
            # self._fillRules(ws)
            #
            # endTime = time.time()
            #
            # self.fitTime = endTime - startTime

            # code for RUG2
            if (len(self.cols) != 0):
                self._cleanup()

            startTime = time.time()

            treeno = 0
            DT = DecisionTreeRegressor(max_depth=self.maxDepth,
                                        random_state=self.randomState)


            fitTree = DT.fit(X, y)
            self.fittedDTs[treeno] = copy.deepcopy(fitTree)
            self._preprocess(X, y)
            self._getInitMatrix(X, y, fitTree, treeno)
            # TODO: Her seferinde yeni model kurmadan da olmalı aslında,
            ws, vs, betas, gammas = self._solvePrimal(y=y)

            self._fillRules(ws)
            yhat = self.predict(X)

            #new CG weights
            newDual = abs(betas+gammas)

            for cg in range(self.maxRMPcalls):
                treeno += 1
                fitTree = DT.fit(X, abs(y-yhat), sample_weight=newDual)
                self.fittedDTs[treeno] = copy.deepcopy(fitTree)
                noImprovement = self._PSPDT(X, y, fitTree, treeno, betas, gammas)
                if (noImprovement):
                    break
                ws, vs, betas, gammas = self._solvePrimal(y, ws0=ws, vs0=vs)
                newDual = abs(betas+gammas)
                self._fillRules(ws)
                yhat = self.predict(X)
            self._fillRules(ws)

            endTime = time.time()

            self.fitTime = endTime - startTime
