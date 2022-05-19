#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18 April 2022

@author: Junhee Lee 496160
"""
#Use this class to iterate the code multiple times and to obtain the average results and standard deviations.
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from rulediscovery import RUXClassifier, RUGClassifier
import Datasets as DS

Number_of_iteration = 100
iterations = np.ones(Number_of_iteration)

randomState = 9
maxDepth = 2
rhsEps = 0.01

# NOTE: Choose Gurobi for large problems like skinnoskin
solver = 'gurobi' # 'glpk' or 'gurobi'
    
problems =[DS.abalone, DS.airfoil, DS.concrete, DS.garments, DS.hitters, DS.powerplant, DS.bike, DS.redwine, DS.whitewine]

for problem in problems:
    pname = problem.__name__.upper()
    print(pname)
    df_orig = np.array(problem('datasets/'))

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df_orig)

    X = df[:, 0:-1]
    y = df[:, -1]

    MAE_RF = np.empty(shape=(0), dtype=float)
    numRule_RF = np.empty(shape=(0), dtype=float)
    MAE_RUXRF = np.empty(shape=(0), dtype=float)
    numRule_RUXRF = np.empty(shape=(0), dtype=float)
    lengthRule_RUXRF = np.empty(shape=(0), dtype=float)
    MAE_ADA = np.empty(shape=(0), dtype=float)
    numRule_ADA = np.empty(shape=(0), dtype=float)
    MAE_RUXADA = np.empty(shape=(0), dtype=float)
    numRule_RUXADA = np.empty(shape=(0), dtype=float)
    lengthRule_RUXADA = np.empty(shape=(0), dtype=float)
    MAE_DT = np.empty(shape=(0), dtype=float)
    numRule_DT = np.empty(shape=(0), dtype=float)
    MAE_RUG = np.empty(shape=(0), dtype=float)
    numRule_RUG = np.empty(shape=(0), dtype=float)
    lengthRule_RUG = np.empty(shape=(0), dtype=float)

    for i in iterations:
        randomState += 1
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

        MAE_RF = np.append(MAE_RF, mean_absolute_error(RF_pred, y_test))
        numRule_RF = np.append(numRule_RF, RUXRF.getInitNumOfRules())
        MAE_RUXRF = np.append(MAE_RUXRF, mean_absolute_error(RUXRF_pred, y_test))
        numRule_RUXRF = np.append(numRule_RUXRF, RUXRF.getNumOfRules())
        lengthRule_RUXRF = np.append(lengthRule_RUXRF, RUXRF.getAvgRuleLength())

        MAE_ADA = np.append(MAE_ADA, mean_absolute_error(ADA_pred, y_test))
        numRule_ADA = np.append(numRule_ADA, RUXADA.getInitNumOfRules())
        MAE_RUXADA = np.append(MAE_RUXADA, mean_absolute_error(RUXADA_pred, y_test))
        numRule_RUXADA = np.append(numRule_RUXADA, RUXADA.getNumOfRules())
        lengthRule_RUXADA = np.append(lengthRule_RUXADA, RUXADA.getAvgRuleLength())

        MAE_DT = np.append(MAE_DT, mean_absolute_error(DT_pred, y_test))
        numRule_DT = np.append(numRule_DT, DT.get_n_leaves())
        MAE_RUG = np.append(MAE_RUG, mean_absolute_error(RUG_pred, y_test))
        numRule_RUG = np.append(numRule_RUG, RUG.getNumOfRules())
        lengthRule_RUG = np.append(lengthRule_RUG, RUG.getAvgRuleLength())

    print('\n\n#### MULTIPLE RUN RESULTS #### \n')
    print('Mean absolute error of RF: ', np.average(MAE_RF), 'standard deviation: ', np.std(MAE_RF))
    print('numRule_RF: ', np.average(numRule_RF), 'standard deviation: ', np.std(numRule_RF))
    print('MAE_RUXRF: ', np.average(MAE_RUXRF), 'standard deviation: ', np.std(MAE_RUXRF))
    print('numRule_RUXRF: ', np.average(numRule_RUXRF), 'standard deviation: ', np.std(numRule_RUXRF))
    print('lengthRule_RUXRF: ', np.average(lengthRule_RUXRF), 'standard deviation: ', np.std(lengthRule_RUXRF))

    print('Mean absolute error of ADA: ', np.average(MAE_ADA), 'standard deviation: ', np.std(MAE_ADA))
    print('numRule_ADA: ', np.average(numRule_ADA), 'standard deviation: ', np.std(numRule_ADA))
    print('MAE_RUXADA: ', np.average(MAE_RUXADA), 'standard deviation: ', np.std(MAE_RUXADA))
    print('numRule_RUXADA: ', np.average(numRule_RUXADA), 'standard deviation: ', np.std(numRule_RUXADA))
    print('lengthRule_RUXADA: ', np.average(lengthRule_RUXADA), 'standard deviation: ', np.std(lengthRule_RUXADA))

    print('MAE_DT: ', np.average(MAE_DT), 'standard deviation: ', np.std(MAE_DT))
    print('numRule_DT: ', np.average(numRule_DT), 'standard deviation: ', np.std(numRule_DT))
    print('MAE_RUG: ', np.average(MAE_RUG), 'standard deviation: ', np.std(MAE_RUG))
    print('numRule_RUG: ', np.average(numRule_RUG), 'standard deviation: ', np.std(numRule_RUG))
    print('lengthRule_RUG: ', np.average(lengthRule_RUG), 'standard deviation: ', np.std(lengthRule_RUG))