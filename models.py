import pandas as pd
import numpy as np
import preprocessingFuncts as pp
from sklearn import neighbors
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import dataAnalysisFuncts as DA

class CF:
  '''
    Constants :
      k-neighbors
      threshold
      metric ('pearson', 'jaccard' or 'kendall')
      TODO - base ('item' or 'user')
    Reusable data:
      Data DataFrame
      Data Matrix
      Similarity Matrix
  '''

  def __init__(self,base,k=5,threshold=30,metric="pearson") -> None:
    if (metric != 'pearson' and metric!= 'kendall' and metric != 'jaccard'):
      raise Exception(f"metric can only be 'pearson', 'kendall' or 'jaccard'; was given {metric}")
    if (base != 'item' and base!='user'):
      raise Exception(
          f"base can only be 'item' or 'user'; was given {base}")
    self.k = k
    self.base = 'user_id' if base == 'user' else 'item_id'
    self.not_base = 'user_id' if base != 'user' else 'item_id'
    self.threshold = threshold
    self.metric = metric
    self.dataDF = 0
    self.dataMatrix = 0
    self.simMatrix = 0
    
  def fit(self,path):
    ratingData = pp.readRatingData(path)

    if(self.dataDF!= 0):
      if(self.dataDF.columns != ratingData.columns):
        raise Exception(f"Columns of inputted data {ratingData.columns} does not match the pre existing data {self.dataDF.columns}")
      self.dataDF.append(ratingData)
    else:
      self.dataDF = ratingData
    
    tempDataMatrix = self.dataDF.pivot_table(
        index=self.not_base, columns=self.base, values='rating')
    print(tempDataMatrix)
    tempDataMatrix = tempDataMatrix.set_axis(
      [int(x) for x in tempDataMatrix.columns], axis='columns', inplace=False)
    tempDataMatrix = tempDataMatrix.set_axis(
      [int(x) for x in tempDataMatrix.index], axis='index', inplace=False)
    
    self.dataMatrix = tempDataMatrix
    self.simMatrix = self.dataMatrix.corr(min_periods=self.threshold)
  
  def predict(self, x):
    y = []
    # print(x.columns)
    # print()
    for _x in x.values:
      # print(_x)
      # print(self.base)
      # raise("debug lol")
      notBaseID, baseID = 0,0
      if(self.base == 'item_id'):
        notBaseID, baseID = _x[0], _x[1]
      else:
        notBaseID, baseID = _x[1], _x[0]

      try:
        simItemIds = self.simMatrix.loc[:,baseID].sort_values(ascending=False)
      except:
        # TODO - figure out how to get unrated movie's sim
        # This try except is made because movie 1582 has never been rated before
        y.append(0)
        continue
      simItemIds = simItemIds.drop(baseID).to_frame().dropna().reset_index().set_axis([self.base,'corr'],axis='columns')
      if (len(simItemIds.index)==0):
        y.append(0)
        continue
      
      _y,a,b = 0,0,0
      _k = self.k
      _idx = 0

      while _k>0 and _idx<len(simItemIds.index):
        try:
          # Because of missing movie 1582
          if(not pd.isna(self.dataMatrix.loc[notBaseID,simItemIds.loc[_idx,self.base]])):
            tempA = simItemIds.loc[_idx,'corr']*self.dataMatrix.loc[notBaseID,simItemIds.loc[_idx,self.base]]
            tempB = simItemIds.loc[_idx,'corr']
            a+=tempA
            b+=tempB
            _k-=1
          _idx += 1
        except:
          break
      try:
        #TODO - fix bug. for some reason some movies' ratings are -10,
        # some others are 6. So bad loll
        # also need to handle the datas that are 0.
        _y = round(a/b) if round(a/b)>0 else 0
        _y = 5 if _y>5 else _y
      except:
        _y = 0
      y.append(_y)
    print(y)
    return y

class SimCF:
  '''
    Constants :
      k-neighbors
      threshold
      metric ('pearson', 'jaccard' or 'kendall')
      TODO - base ('item' or 'user')
    Reusable data:
      Data DataFrame
      Data Matrix
      Similarity Matrix
  '''

  def __init__(self,base,k=5,threshold=30,metric="pearson") -> None:
    if (metric != 'pearson' and metric!= 'kendall' and metric != 'jaccard'):
      raise Exception(f"metric can only be 'pearson', 'kendall' or 'jaccard'; was given {metric}")
    if (base != 'item' and base!='user'):
      raise Exception(
          f"base can only be 'item' or 'user'; was given {base}")
    self.k = k
    self.base = 'user_id' if base == 'user' else 'item_id'
    self.not_base = 'user_id' if base != 'user' else 'item_id'
    self.threshold = threshold
    self.metric = metric
    self.dataDF = 0
    self.dataMatrix = 0
    self.simMatrix = 0
    
  def fit(self,path):
    ratingData = pp.readRatingData(path)

    if(self.dataDF!= 0):
      if(self.dataDF.columns != ratingData.columns):
        raise Exception(f"Columns of inputted data {ratingData.columns} does not match the pre existing data {self.dataDF.columns}")
      self.dataDF.append(ratingData)
    else:
      self.dataDF = ratingData
    
    #TODO - do this, but to user_id and item_id, not just for rating
    tempDataMatrix = self.dataDF.pivot_table(
        index=self.not_base, columns=self.base, values='rating')
    tempDataMatrix = tempDataMatrix.set_axis(
      [int(x) for x in tempDataMatrix.columns], axis='columns', inplace=False)
    tempDataMatrix = tempDataMatrix.set_axis(
      [int(x) for x in tempDataMatrix.index], axis='index', inplace=False)
    print(tempDataMatrix)
    
    self.dataMatrix = tempDataMatrix
    self.simMatrix = self.dataMatrix.corr(min_periods=self.threshold)

  
  def predict(self, x):
    y = []
    for _x in x.values:
      notBaseID, baseID = (_x[0], _x[1]) if self.base=='item' else (_x[1], _x[0])
      try:
        simItemIds = self.simMatrix.loc[:,baseID].sort_values(ascending=False)
      except:
        # TODO - figure out how to get unrated movie's sim
        # This try except is made because movie 1582 has never been rated before
        y.append(0)
        continue

      simItemIds = simItemIds.drop(baseID).to_frame().dropna().reset_index().set_axis([self.base,'corr'],axis='columns')
      
      if (len(simItemIds.index)==0):
        y.append(0)
        continue
      
      _y,a,b = 0,0,0
      _k = self.k
      _idx = 0

      while _k>0 and _idx<len(simItemIds.index):
        if(not pd.isna(self.dataMatrix.loc[notBaseID,simItemIds.loc[_idx,self.base]])):
          tempA = simItemIds.loc[_idx,'corr']*self.dataMatrix.loc[notBaseID,simItemIds.loc[_idx,self.base]]
          tempB = simItemIds.loc[_idx,'corr']
          a+=tempA
          b+=tempB
          _k-=1
        _idx += 1
      try:
        #TODO - fix bug. for some reason some movies' ratings are -10,
        # some others are 6. So bad loll
        # also need to handle the datas that are 0.
        _y = round(a/b) if round(a/b)>0 else 0
        _y = 5 if _y>5 else _y
      except:
        _y = 0
      y.append(_y)
    print(y)
    return y

#TODO - Similarity Model for SimCF
IBCF = SimCF('item')
IBCF.fit('ml-100k\\ua.base')




