import preprocessingFuncts as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def specifyByItemData(items, ratings, categ):
    # categ only 2, genres or year
    item_header = ["item_id"]
    if categ == "year":
        item_header.append("year")
    elif categ == "genres":
        item_header.extend(["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    elif categ == "all":
        item_header.append("year")
        item_header.extend(["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    else:
        raise Exception(
            "category can only be strings \"year\", \"genres\" or \"all\"")
    # print(item_header)
    # display(items)
    _item = items.loc[:, item_header]
    df = pd.merge(_item, ratings, on=['item_id'])
    return df

def categorySimilarity(occup,arr,tick,string,size=(20,20)):
    sim = occup.pivot_table(columns=string,index='item_id',values='rating') #Get the pivot table
    a = sim.corr(min_periods=30) #Get the correlation with threshold 30
    plt.figure(figsize=size) #figure the size, default = 20,20, we can set it based on what we like
    plt.set_cmap('jet') #Set the color of the box
    plt.imshow(a) #Create the matrix table
    plt.colorbar() #Show the right side color
    # print(a)
    for i in a.columns:
        # print('a')
        for j in a.columns:
            # print(j+" "+i)
            plt.text(i,j,str(a[j][i].round(2)),va='center',ha='center') #setting the text in each matrix box
    plt.xticks(range(0,len(tick)),tick,rotation=-90) 
    plt.yticks(range(0,len(tick)),tick)
    plt.show() #show the plot

items = pp.readItemData()
items = items.sort_values(by=['year','item_id']).reset_index().drop('index',axis=1)
items = items.dropna()

ratings = pp.readRatingData()
ratings = ratings.sort_values(by=['user_id','item_id']).reset_index().drop('index',axis=1)
ratings

occup = saveyear = pp.specifyByItemData(items, ratings, "year")
occup = occup.drop('user_id',axis=1)
occup = occup.groupby(by=['year','item_id']).mean()

#saving only the year list
saveyear = saveyear['year'].drop_duplicates().reset_index().drop('index',axis=1)
# display(saveyear)
saveyeartext = saveyear['year'].tolist()
# display(saveyeartext)
categorySimilarity(occup,saveyear,saveyeartext,'year')
