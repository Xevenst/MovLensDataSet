import pandas as pd
import numpy as np
import preprocessingFuncts as pp
from sklearn import neighbors
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

class ItemBasedCF:
  '''
    TODO - WRITE CLASS INFO HERE
  '''
  def __init__(self,k,metric) -> None:
    if (metric != 'pearson' or metric!= 'kendall' or metric != 'jaccard'):
      raise Exception(f"metric can only be 'pearson', 'kendall' or 'jaccard'; was given {metric}")
    self.metric = metric
    self.k = k
    self.matrix = 0
    
  def fit(self,path):
    if(self.matrix == 0):
      ratingData = pp.readRatingData(path)
      UI = ratingData.pivot_table(
          index='user_id', columns='item_id', values='rating')
      UI = UI.set_axis([int(x) for x in UI.columns], axis='columns', inplace=False)
      UI = UI.set_axis([int(x) for x in UI.index], axis='index', inplace=False)
      self.matrix = UI
    else:
      
  