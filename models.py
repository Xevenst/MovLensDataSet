import pandas as pd
import numpy as np
import preprocessingFuncts as pp
from sklearn import neighbors
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

class ItemBasedCF:
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

  def __init__(self,k=5,threshold=30,metric="pearson") -> None:
    if (metric != 'pearson' and metric!= 'kendall' and metric != 'jaccard'):
      raise Exception(f"metric can only be 'pearson', 'kendall' or 'jaccard'; was given {metric}")
    self.k = k
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
        index='user_id', columns='item_id', values='rating')
    tempDataMatrix = tempDataMatrix.set_axis(
      [int(x) for x in tempDataMatrix.columns], axis='columns', inplace=False)
    tempDataMatrix = tempDataMatrix.set_axis(
      [int(x) for x in tempDataMatrix.index], axis='index', inplace=False)

    self.dataMatrix = tempDataMatrix
    self.simMatrix = self.dataMatrix.corr(min_periods=self.threshold)
  
  def predict(self, x):
    y = []
    for _x in x.values:
      uID, iID = _x[0], _x[1]
      try:
        simItemIds = self.simMatrix.loc[:,iID].sort_values(ascending=False)
      except:
        # TODO - figure out how to get unrated movie's sim
        # This try except is made because movie 1582 has never been rated before
        y.append(0)
        continue

      simItemIds = simItemIds.drop(iID).to_frame().dropna().reset_index().set_axis(['item_id','corr'],axis='columns')
      
      if (len(simItemIds.index)==0):
        y.append(0)
        continue
      
      _y,a,b = 0,0,0
      _k = self.k
      _idx = 0

      while _k>0 and _idx<len(simItemIds.index):
        if(not pd.isna(self.dataMatrix.loc[uID,simItemIds.loc[_idx,'item_id']])):
          a+=simItemIds.loc[_idx,'corr']*self.dataMatrix.loc[uID,simItemIds.loc[_idx,'item_id']]
          b+=simItemIds.loc[_idx,'corr']
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
    return y





