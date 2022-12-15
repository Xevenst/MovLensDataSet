import pandas as pd
import numpy as np
import preprocessingFuncts as pp
from sklearn import neighbors


'''''''''
TODO 
- take simi for rating,occup,age,gender
- weighted parameter [made rating most important ,second to occup and age,gender atleast signi]
- 

'''''''''
class  simiModel:
      def __init__(self,base,k=5,threshold=30) -> None:
        self.k = k
        self.base = 'user_id' if base == 'user' else 'item_id'
        self.not_base = 'user_id' if base != 'user' else 'item_id'
        self.threshold = threshold
        self.dataDF = 0
        self.dataMatrix = 0
        self.simMatrix = 0