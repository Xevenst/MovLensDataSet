import pandas as pd
import numpy as np
import preprocessingFuncts as pp
import math
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
        self.combMatrix=0
        self.weigth= weigth
        if base=="user":
            self.length= "user_id"
        else:
            self.length= "item_id"

       
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
        # print(self.simMatrix)
        # print(self.base)
        # print(len(tempDataMatrix.columns))
        # self.combMatrix=[0.0*len(self.simMatrix)]*len(self.simMatrix)
        self.combMatrix = np.zeros((len(self.simMatrix),len(self.simMatrix)))
        print(self.simMatrix.fillna(0))
        filter=(self.dataDF[self.length]==2)
        tempocci=self.dataDF[filter].iloc[0,4]
        exit()

        for i in range(len(self.simMatrix)):
            temparr=[]
            filter=(self.dataDF[self.length]==i)
            tempocci=self.dataDF[filter].iloc[0,4]
            tempagei=self.dataDF[filter].iloc[0,5]
            tempgeni=self.dataDF[filter].iloc[0,3]
            print(f'this is {i} ={tempocci,tempagei,tempgeni}')
            for j in range(len(self.simMatrix)):
                print(i,j)
                print(self.simMatrix[i][j])
                exit()
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
                    self.weigth[0]=0.0
                else:
                    temparr.append(tempsimrate)

                if math.isnan(tempsimocc):
                    temparr.append(0.0)
                elif tempsimocc == 0.0:
                    self.weigth[1]=0.0
                else:
                    temparr.append(tempsimocc)

                if math.isnan(tempsimage):
                    temparr.append(0.0)
                elif tempsimage == 0.0:
                    self.weigth[2]=0.0
                else:
                    temparr.append(tempsimage)

                if math.isnan(tempsimgen):
                    temparr.append(0.0)
                elif tempsimgen == 0.0:
                    self.weigth[3]=0.0
                else:
                    temparr.append(tempsimgen)
                print(f'This is arr:{temparr}')

                result=(self.weigth[0]*temparr[0]+self.weigth[1]*temparr[1]+self.weigth[2]*temparr[2]+self.weigth[3]*temparr[3])/(self.weigth[0]+self.weigth[1]+self.weigth[2]+self.weigth[3])
                print(f'result: {result} the type:{type(result)}')
                self.combMatrix[i][j]=result
                break
            #print(self.combMatrix)
            # if i not in self.simMatrix.columns:
            #     continue 
            
                
            # print("HE")
        #print(self.combMatrix)
                


    

TEA = simiModel('user')
TEA.fit('ml-100k\\ua.base','ml-100k\\u.user')
