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
'''''''''
CONTINUE TOMMOROW FOR THE ITEM BASE AND USER BASE THOUGH 
'''''''''
class  simiModel:
    def __init__(self,base,k=5,threshold=30,weigth=[5,3,3,1]) -> None:
        self.k = k
        self.base = 'user_id' if base == 'user' else 'item_id'
        self.not_base = 'user_id' if base != 'user' else 'item_id'
        self.threshold = threshold
        self.dataDF = 0
        self.dataMatrix = 0
        self.simMatrix = 0
        self.combMatrix= 0
        self.weigth= weigth
    
       
    def fit(self,path,userdatapath):
        ratingData = pp.readRatingData(path)
        userData = pp.readUserData(userdatapath)
        if(self.dataDF!= 0):
            if(self.dataDF.columns != ratingData.columns):
                raise Exception(f"Columns of inputted data {ratingData.columns} does not match the pre existing data {self.dataDF.columns}")
            self.dataDF.append(ratingData)
        else:
            self.dataDF = ratingData
        
        self.dataDF = pd.merge(self.dataDF,userData)
        self.dataDF = self.dataDF.drop(['age','zip_code'],axis=1)
        simocc= pp.getSimOccup()
        simage= pp.getSimAgeCategory()
        simgen= pp.getSimGender()
        print(self.dataDF)
        #print(len(userData))
        # print(simocc,simage,simgen)

        #print(self.dataDF)
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
        print(self.simMatrix)
        print(self.base)
        # print(len(tempDataMatrix.columns))
        # for i in self.simMatrix:
        #     print(i)
            # tempocci=(self.dataDF["user_id"]==i)
            # tempocci=self.dataDF[tempocci].iloc[0,4]
            # tempagei=(self.dataDF["user_id"]==i)
            # tempagei=self.dataDF[tempagei].iloc[0,5]
            # tempgeni=(self.dataDF["user_id"]==i)
            # tempgeni=self.dataDF[tempgeni].iloc[0,3]
            # tempIdi=(self.dataDF["user_id"]==i)
            # tempIdi=self.dataDF[tempIdi].iloc[0,1]
            #print(f'this is {i} ={tempocci,tempagei,tempgeni}')
        #     for j in self.simMatrix:
        #         tempsimrate=self.dataMatrix[i][j]
        #         tempoccj=(self.dataDF["user_id"]==j)
        #         tempoccj=self.dataDF[tempoccj].iloc[0,4]
        #         tempagej=(self.dataDF["user_id"]==j)
        #         tempagej=self.dataDF[tempagej].iloc[0,5]
        #         tempgenj=(self.dataDF["user_id"]==j)
        #         tempgenj=self.dataDF[tempgenj].iloc[0,3]
        #         tempsimocc=simocc.at[tempocci,tempoccj]
        #         print(tempsimocc,i,j)
                # tempsimage=simage.at[tempagei,tempagej]
                # tempsimgen=simgen.at[tempgeni,tempgenj]
                # temparr= temparr.append(tempsimrate,tempsimocc,tempsimage,tempsimgen)
                # print(temparr)
            # print("HE")
                


    

TEA = simiModel('item')
TEA.fit('ml-100k\\ua.base','ml-100k\\u.user')
