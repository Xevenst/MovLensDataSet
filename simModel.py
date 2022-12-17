import pandas as pd
import numpy as np
import preprocessingFuncts as pp
import math
from sklearn import neighbors
import warnings
warnings.filterwarnings("ignore") #to remove warnings

'''''''''
TODO 
- take simi for rating,occup,age,gender
- weighted parameter [made rating most important ,second to occup and age,gender atleast signi]
- 

'''''''''
'''''''''
CONTINUE TOMMOROW FOR THE ITEM BASE AND USER BASE THOUGH 
'''''''''
class  simiModel:
    def __init__(self,base,k=5,threshold=30,weight=[5,3,3,1]) -> None:
        self.k = k
        self.base = 'user_id' if base == 'user' else 'item_id'
        self.not_base = 'user_id' if base != 'user' else 'item_id'
        self.threshold = threshold
        self.dataDF = 0
        self.dataMatrix = 0
        self.simMatrix = 0
        self.combMatrix=0
        self.weight= weight
        self.simRating = None
        if base=="user":
            self.length= "user_id"
        else:
            self.length= "item_id"

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
        except Exception as e:
          # TODO - figure out how to get unrated movie's sim
          # This try except is made because movie 1582 has never been rated before
          print(e)
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
        except Exception as e:
          print(e)
          _y = 0
        y.append(_y)
      print(y)
      return y

    def fit(self,path):
        ratingData = pp.readRatingData(path)
        userData = pp.readUserData()
        if(self.dataDF!= 0):
            if(self.dataDF.columns != ratingData.columns):
                raise Exception(f"Columns of inputted data {ratingData.columns} does not match the pre existing data {self.dataDF.columns}")
            self.dataDF.append(ratingData)
        else:
            self.dataDF = ratingData
        
        self.dataDF = pd.merge(self.dataDF,userData)
        self.dataDF = self.dataDF.drop(['age','zip_code'],axis=1)
        simocc = self.weight[1]*pp.getSimOccup()
        simage = self.weight[2]*pp.getSimAgeCategory()
        simgen = self.weight[3]*pp.getSimGender()


        userData = userData.set_index('user_id')
        #TODO - do this, but to user_id and item_id, not just for rating
        tempDataMatrix = self.dataDF.pivot_table(
            index=self.not_base, columns=self.base, values='rating')
        tempDataMatrix = tempDataMatrix.set_axis(
        [int(x) for x in tempDataMatrix.columns], axis='columns', inplace=False)
        tempDataMatrix = tempDataMatrix.set_axis(
        [int(x) for x in tempDataMatrix.index], axis='index', inplace=False)
        #print(tempDataMatrix)
        self.dataMatrix = tempDataMatrix
        self.simMatrix = self.dataMatrix.corr(min_periods=self.threshold)
        self.simRating = self.weight[0]*self.simMatrix
        endresult = pd.DataFrame(np.zeros((943, 943)),columns=range(1,944),index=range(1,944))
        # print(userData.loc[0,'age_category'])
        self.simRating = self.simRating.loc[userData.index,userData.index] #Probably wrong but i'll try it when the code finishes

        simage = simage.loc[userData.loc[userData.index,'age_category'],userData.loc[userData.index,'age_category']]
        simage.index = range(1,944)
        simage.columns = range(1,944)
        
        simocc = simocc.loc[userData.loc[userData.index,'occupation'],userData.loc[userData.index,'occupation']]
        simocc.index = range(1,944)
        simocc.columns = range(1,944)

        simgen = simgen.loc[userData.loc[userData.index,'gender'],userData.loc[userData.index,'gender']]
        simgen.index = range(1,944)
        simgen.columns = range(1,944)
        
        pembagian = endresult.copy()
        tempb = (self.simRating.notna()*5) + (simage.notna()*3) + (simocc.notna()*3) +(simgen.notna()*1)
        # ((not self.simRating.isna())*5).add((not simage.isna())*3).add((not simocc.isna())*3).add((not simgen.isna())*3)
        # pembagian.loc[self.simRating.loc[:,:].isna(),:] = 1
        endresult = (self.simRating.fillna(0) + simage + simocc + simgen) / tempb
        self.simMatrix = endresult
        return
        


        # tempsimage = [list(x) for x in]
        for i in range(1,944):
            # pembagian[i] = self.simMatrix[pd.isna(self.simMatrix)]
            # endresult[i] = endresult[i].apply(lambda x:print(x))
            exit()


        print(pembagian)
        #calculating the data sim multiplied by weight and add it to new dataframe
        exit()

        for i in userData.index:
            tempa=0
            tempb=0
            for j in userData.index:
                if(j==i):
                    continue
                #[weight,sim] : [rating,age_category,occupation,gender]
                a=[]
                
                #get the rating similarity and put it to array
                a.append([5,self.simRating.loc[i,j]])

                indexi = userData.loc[i,'age_category']
                indexj = userData.loc[j,'age_category']
                a.append([3,simage.loc[indexi,indexj]])

                indexi = userData.loc[i,'occupation']
                indexj = userData.loc[j,'occupation']
                a.append([3,simocc.loc[indexi,indexj]])

                indexi = userData.loc[i,'gender']
                indexj = userData.loc[j,'gender']
                a.append([1,simgen.loc[indexi,indexj]])

                # print(a)

                #Add tempa and tempb
                for aa in a:
                    # print(aa)
                    if not math.isnan(aa[1]):
                        tempa+=aa[0]*aa[1]
                        # print(f'tempa = {tempa} from {aa[0]} * {aa[1]}')
                        tempb+=aa[0]
                        # print(f'tempb = {tempb}')
            
                tempresult = tempa/tempb
                # print(f'tempresult = {tempresult} from tempa:{tempa}/tempb:{tempb}')
                    # else:
                    #     print("Nan found")
                tempdata = pd.DataFrame({'user_id 1':[i],'user_id 2':[j],'similarity':[tempresult]})
            
                # tempdata.append([i,j,result],columns=['user_id','user_id','similarity']))
                # print(tempdata)
                endresult = endresult.append(tempdata)
            print(endresult)
                
        # 1. siapin dataframe kosong bandingan user ama user
        # 2. loop antar user
        #       i. get the values for dem similarities (rating,occup,age_categ,gender)
        #       ii. calculate the similarity
        #       iii. put in dataframe

        exit()
        # print(self.simMatrix)
        # print(len(tempDataMatrix.columns))
        # self.combMatrix=[0.0*len(self.simMatrix)]*len(self.simMatrix)
        # self.combMatrix = np.zeros((len(self.simMatrix),len(self.simMatrix)))
        # print(self.simMatrix.fillna(0))
        filter=()
        tempocci=self.dataDF[filter].iloc[0,4]

        for i in int(self.simMatrix.index.name):
            print(i)
            temparr=[]
            filter=(self.dataDF[self.length]==i)
            tempocci=self.dataDF[self.dataDF[self.length]==2].iloc[0,4]
            tempagei=self.dataDF[filter].iloc[0,5]
            tempgeni=self.dataDF[filter].iloc[0,3]
            print(f'this is {i} ={tempocci,tempagei,tempgeni}')
            for j in int(self.simMatrix.index.name):
                print(i,j)
                print(self.simMatrix[i][j])
                # if j not in self.simMatrix.columns:
                #     print("YES")
                # else:
                #     print("NO")
                tempsimrate=self.simMatrix[i][j] 
                print(f'This is tempsimrate {i,j}: {tempsimrate}')
                filter=(self.dataDF[self.length]==j)
                tempoccj=self.dataDF[filter].iloc[0,4]
                tempagej=self.dataDF[filter].iloc[0,5]
                tempgenj=self.dataDF[filter].iloc[0,3]
                tempsimocc=simocc.at[tempocci,tempoccj]
                tempsimage=simage.at[tempagei,tempagej]
                tempsimgen=simgen.at[tempgeni,tempgenj]
                if math.isnan(tempsimrate):
                    temparr.append(0.0)
                elif tempsimrate== 0.0:
                    self.weight[0]=0.0
                else:
                    temparr.append(tempsimrate)

                if math.isnan(tempsimocc):
                    temparr.append(0.0)
                elif tempsimocc == 0.0:
                    self.weight[1]=0.0
                else:
                    temparr.append(tempsimocc)

                if math.isnan(tempsimage):
                    temparr.append(0.0)
                elif tempsimage == 0.0:
                    self.weight[2]=0.0
                else:
                    temparr.append(tempsimage)

                if math.isnan(tempsimgen):
                    temparr.append(0.0)
                elif tempsimgen == 0.0:
                    self.weight[3]=0.0
                else:
                    temparr.append(tempsimgen)
                print(f'This is arr:{temparr}')

                result=(self.weight[0]*temparr[0]+self.weight[1]*temparr[1]+self.weight[2]*temparr[2]+self.weight[3]*temparr[3])/(self.weight[0]+self.weight[1]+self.weight[2]+self.weight[3])
                print(f'result: {result} the type:{type(result)}')
                self.combMatrix[i][j]=result
                break
            #print(self.combMatrix)
            # if i not in self.simMatrix.columns:
            #     continue 
            
                
            # print("HE")
        #print(self.combMatrix)
                
TEA = simiModel('user')
TEA.fit('ml-100k\\ua.base')

testData = pp.readRatingData('ml-100k\\ua.test')
testX, testY =  testData.loc[:,['user_id','item_id']],testData.loc[:,'rating']

predY = TEA.predict(testX)

from sklearn.metrics import classification_report

print(classification_report(testY, predY))
