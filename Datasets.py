#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18 April 2022

@author: Junhee Lee 496160
"""
import pandas as pd

def abalone(wd):
    df = pd.read_csv(wd+'abalone.csv', header = None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns)-1)] + ['y']
    return df


def airfoil(wd):
    df = pd.read_csv(wd + 'airfoil_self_noise.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df


def concrete(wd):
    df = pd.read_csv(wd + 'Concrete_Data.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df


def garments(wd):
    df = pd.read_csv(wd + 'garments_worker_productivity.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def hitters(wd):
    df = pd.read_csv(wd + 'Hitters.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def powerplant(wd):
    df = pd.read_csv(wd + 'power_plant.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def bike(wd):
    df = pd.read_csv(wd + 'SeoulBikeData.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def redwine(wd):
    df = pd.read_csv(wd + 'winequality-red.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df

def whitewine(wd):
    df = pd.read_csv(wd + 'winequality-white.csv', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    return df
