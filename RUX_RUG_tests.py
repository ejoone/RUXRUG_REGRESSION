#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18 April 2022

@original author: sibirbil
@author: Junhee Lee 496160
"""
# Use this class to obtain the result with detailed information about the rules of RUX and RUG.
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from rulediscovery import RUXClassifier, RUGClassifier
import Datasets as DS

randomState = 9
maxDepth = 2
rhsEps = 0.01

# NOTE: Choose Gurobi for large problems like skinnoskin
solver = 'gurobi' # 'glpk' or 'gurobi'
    
problems =[DS.abalone, DS.airfoil, DS.concrete, DS.garments, DS.hitters, DS.powerplant, DS.bike, DS.redwine, DS.whitewine]
problems = [DS.airfoil]

for problem in problems:
    pname = problem.__name__.upper()
    print(pname)

    df_orig = np.array(problem('datasets/'))

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df_orig)

    X = df[:, 0:-1]
    y = df[:, -1]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=randomState, test_size=0.3)

    RF = RandomForestRegressor(max_depth=maxDepth, random_state=randomState)
    RF_fit = RF.fit(X_train, y_train)
    RF_pred = RF_fit.predict(X_test)


    RUXRF = RUXClassifier(rf=RF_fit, eps=rhsEps,
                          rule_length_cost=True,
                          false_negative_cost=False,
                          solver=solver,
                          random_state=randomState)
    RUXRF_fit = RUXRF.fit(X_train, y_train)
    RUXRF_pred = RUXRF.predict(X_test)


    ADA = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=maxDepth),
                              loss='linear',
                              random_state=randomState)
    ADA_fit = ADA.fit(X_train, y_train)
    ADA_pred = ADA_fit.predict(X_test)


    RUXADA = RUXClassifier(ada=ADA_fit, eps=rhsEps,
                            use_ada_weights=True,
                            solver=solver,
                            random_state=randomState)
    RUXADA_fit = RUXADA.fit(X_train, y_train)
    RUXADA_pred = RUXADA.predict(X_test)


    RUG = RUGClassifier(eps=rhsEps,
                        max_depth=maxDepth,
                        rule_length_cost=True,
                        false_negative_cost=False,
                        solver=solver,
                        random_state=randomState)
    RUG_fit = RUG.fit(X_train, y_train)
    RUG_pred = RUG.predict(X_test)

    DT = DecisionTreeRegressor(max_depth=maxDepth,
                               random_state=randomState)
    DT_fit=DT.fit(X_train,y_train)
    DT_pred=DT.predict(X_test)




    print('\n\n#### RESULTS #### \n')
    print('Mean absolute error of RF: ', mean_absolute_error(RF_pred, y_test))

    print('Mean absolute error of RUX(RF): ', mean_absolute_error(RUXRF_pred, y_test))
    print('Total number of RF rules: ', RUXRF.getInitNumOfRules())
    print('Total number of rules in RUX(RF): ', RUXRF.getNumOfRules())
    print('Total number of missed samples in RUX(RF): ', RUXRF.getNumOfMissed())
    print('Training time for RUX(RF)', RUXRF.getFitTime())
    print('Prediction time for RUX(RF)', RUXRF.getPredictTime())
    print('avg rule length', RUXRF.getAvgRuleLength())
    RUXRF.printRules()

    print('Mean absolute error of ADA: ', mean_absolute_error(ADA_pred, y_test))

    print('Mean absolute error of RUX(ADA): ', mean_absolute_error(RUXADA_pred, y_test))
    print('Total number of ADA rules: ', RUXADA.getInitNumOfRules())
    print('Total number of rules in RUX(ADA): ', RUXADA.getNumOfRules())
    print('Total number of missed samples in RUX(ADA): ', RUXADA.getNumOfMissed())
    print('Training time for RUX(ADA)', RUXADA.getFitTime())
    print('Prediction time for RUX(ADA)', RUXADA.getPredictTime())
    print('avg rule length', RUXADA.getAvgRuleLength())
    RUXADA.printRules()

    print('Mean absolute error of RUG: ', mean_absolute_error(RUG_pred, y_test))
    print('Total number of rules in RUG: ', RUG.getNumOfRules())
    print('Total number of missed samples in RUG: ', RUG.getNumOfMissed())
    print('Training time for RUG', RUG.getFitTime())
    print('Prediction time for RUG', RUG.getPredictTime())
    print('avg rule length', RUG.getAvgRuleLength())
    RUG.printRules()

    print('Mean absolute error of DT: ', mean_absolute_error(DT_pred, y_test))
    print('Total number of DT rules: ', DT.get_n_leaves())
