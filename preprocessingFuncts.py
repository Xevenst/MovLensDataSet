import pandas as pd
import numpy as np
from IPython.display import display, HTML

# use this file for functions to read data from the ml- 100k file
# I just made this to separate data and throw away useless ones

'''
mainly, data to analyze from u.data, u.user, u.item
u.data  -   the ratings data. user rated item
            [userID|itemID|Rating|timestamp]
u.item  -   The information regarding the movies
            [movie id | movie title | release date | video release date | IMDb URL | ..genres 19 fields.. ]
u.user  -   Info about the users
            [user id | age | gender | occupation | zip code ]
'''

'''
u.occupation is to accompany u.user
u.genre is to accompany u.item
'''

# first, read the rating data


def readRatingData(path="ml-100k\\u.data"):
    rating_header = ["user_id", "item_id", "rating", "timestamp"]
    rating = pd.read_csv(path, sep='\t',
                         header=None, names=rating_header)
    rating = rating.drop(['timestamp'], axis=1)
    return rating


def readItemData(path="ml-100k\\u.item"):
    movie_header = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movies = pd.read_csv(path, sep='|',
                         header=None, encoding='latin1', names=movie_header)
    movies["release_date"] = movies["release_date"].map(
        lambda x: x[-4:] if type(x) == str else x)
    # the only columns that matter is just id and genres hahahaah
    movies = movies.drop(
        columns=['video_release_date', "release_date", "IMDb_URL"])
    movies = movies.rename(columns={"release_date": "year"})
    return movies


def readUserData(path="ml-100k\\u.user"):
    user_header = ["user_id", "age", "gender", "occupation", "zip_code"]
    users = pd.read_csv(path, sep='|',
                        header=None, names=user_header)

    occupation = pd.read_csv("ml-100k\\u.occupation", header=None)
    occupation_list = occupation.values

    

    users["gender"].replace(['F', 'M'], [0, 1], inplace=True)
    users["occupation"].replace(occupation_list, list(
        range(0, len(occupation_list))), inplace=True)
    users["age_category"] = pd.cut(users["age"], bins = [0, 10, 20, 30, 40, 50, 60, 70, 80], labels=[1, 2, 3, 4, 5, 6, 7, 8 ])
    #print(users["age_category"])
    return users



# Best/worst ratings for user categs
def Unweighteduserdata(categ):
    rating=readRatingData()
    users=readUserData()
    movies=readItemData()
    average_rating_baseonI= rating[["item_id", "rating"]].groupby(["item_id"], as_index=False).mean() # average rating per movie
    average_rating_baseonI.rename(columns = {'rating':'average_rating'}, inplace = True)
    rating=pd.merge(rating,average_rating_baseonI)

    rating=pd.merge(rating,users[["user_id","gender","occupation","age_category"]])
    rating=pd.merge(rating,movies[["item_id","title"]])
    rating=rating.drop(["user_id","rating"],axis=1)
    storedparameter=[]
    if categ=="gender":
    ######################Gender######################
        for a in range(2):
            parameterMax=rating[rating["gender"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["gender"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)
            #print(f"Gender {a}:")
            # display(parameterMax)
            # display(parameterMin)
            # display(storedparameter)
    ##################################################
    elif categ=="occupation":
    ######################occupation##################
        a=0
        for a in range(21):
            parameterMax=rating[rating["occupation"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["occupation"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)
            # print(f"Occupation {a}:")
            # display(parameterMax)
            # display(parameterMin)
    ##################################################
    elif categ=="age_group":
    ######################Age_group###################
        a=0
        for a in range(9):
            parameterMax=rating[rating["age_category"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["age_category"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)
            # print(f"Age_group {a}:")
            # display(parameterMax)
            # display(parameterMin)
        else:
            raise Exception(f"categ should be 'gender' or 'occupation' or 'age_group'; given {categ}")
    ##################################################
    #print(gendercontainerMax)
    #print(storedparameter)
    return storedparameter



def Weighteduserdata(threshold, categ):
    rating=readRatingData()
    users=readUserData()
    movies=readItemData()
    #print(rating.sort_values(by=["user_id","item_id"],ascending=True))
    average_rating_baseonI= rating[["item_id", "rating"]].groupby(["item_id"], as_index=False).mean() # average rating per movie
    average_rating_baseonI.rename(columns = {'rating':'average_rating'}, inplace = True)
    #print(average_rating_baseonI.sort_values(by=["item_id"],ascending=False))
    rating=pd.merge(rating,average_rating_baseonI)
    weight=pd.DataFrame()
    weight["count"]=rating.groupby(["item_id"])["item_id"].count()
    weight=weight.reset_index()
    
    filter=(weight["count"]>=threshold)
    weight=weight[filter]
    rating=pd.merge(rating,weight).sort_values(by=["count"],ascending=True)
    rating=pd.merge(rating,users[["user_id","gender","occupation","age_category"]])
    rating=pd.merge(rating,movies[["item_id","title"]])
    rating=rating.drop(["user_id","rating","count"],axis=1)
    #print(rating)
    parameterMax=0
    parameterMin=0
    storedparameter=[]
    if categ=="gender":
    ######################Gender######################
        for a in range(2):
            parameterMax=rating[rating["gender"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["gender"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)
            
            #print(f"Gender {a}:")
            # display(parameterMax)
            # display(parameterMin)
            # display(storedparameter)
    ##################################################
    elif categ=="occupation":
    ######################occupation##################
        a=0
        for a in range(21):
            parameterMax=rating[rating["occupation"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["occupation"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)
            # print(f"Occupation {a}:")
            # display(parameterMax)
            # display(parameterMin)
    ##################################################
    elif categ=="age_group":
    ######################Age_group###################
        a=0
        for a in range(9):
            parameterMax=rating[rating["age_category"]==a].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            parameterMin=rating[rating["age_category"]==a].groupby("average_rating").min().sort_values("average_rating",ascending=True).head()
            parameterMax=parameterMax.drop(["gender","occupation","age_category"],axis=1)
            parameterMin=parameterMin.drop(["gender","occupation","age_category"],axis=1)
            storedparameter.append(parameterMax)
            storedparameter.append(parameterMin)
            # print(f"Age_group {a}:")
            # display(parameterMax)
            # display(parameterMin)
        else:
            raise Exception(f"categ should be 'gender' or 'occupation' or 'age_group'; given {categ}")
    ##################################################
    #print(gendercontainerMax)
    #print(storedparameter)
    return storedparameter
#Weighteduserdata(30,"gender")
#Unweighteduserdata("gender")

def Weighteditemdata(threshold,moviegenre):
    rating=readRatingData()
    movies=readItemData()
    average_rating_baseonI= rating[["item_id", "rating"]].groupby(["item_id"], as_index=False).mean() # average rating per movie
    average_rating_baseonI.rename(columns = {'rating':'average_rating'}, inplace = True)
    rating=pd.merge(rating,average_rating_baseonI)
    weight=pd.DataFrame()
    weight["count"]=rating.groupby(["item_id"])["item_id"].count()
    weight=weight.reset_index()
    filter=(weight["count"]>=threshold)
    weight=weight[filter]
    rating=pd.merge(rating,weight).sort_values(by=["count"],ascending=True)
    rating=pd.merge(rating,movies)
    rating=rating.drop(["user_id","rating","count"],axis=1)
    moviedict=["unknown", "Action", "Adventure", "Animation", "Children's",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    final=[]
    if moviegenre in moviedict:
        if moviegenre == "unknown":
            raise Exception("Weighted data for Unknown genre does not exist")
        else:
            testmax=rating[rating[moviegenre]==1].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            testmax=testmax[["item_id","title"]]
            final.append(testmax)
            testmin=rating[rating[moviegenre]==1].groupby("average_rating").max().sort_values("average_rating",ascending=True).head()
            testmin=testmin[["item_id","title"]]
            final.append(testmin)
    else:
        raise Exception("rewrite the genre")
    return final

def Unweighteditemdata(moviegenre):
    rating=readRatingData()
    movies=readItemData()
    average_rating_baseonI= rating[["item_id", "rating"]].groupby(["item_id"], as_index=False).mean() # average rating per movie
    average_rating_baseonI.rename(columns = {'rating':'average_rating'}, inplace = True)
    rating=pd.merge(rating,average_rating_baseonI)
    rating=pd.merge(rating,movies)
    rating=rating.drop(["user_id","rating"],axis=1)
    moviedict=["unknown", "Action", "Adventure", "Animation", "Children's",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    final=[]
    if moviegenre in moviedict:
            testmax=rating[rating[moviegenre]==1].groupby("average_rating").max().sort_values("average_rating",ascending=False).head()
            testmax=testmax[["item_id","title"]]
            final.append(testmax)
            testmin=rating[rating[moviegenre]==1].groupby("average_rating").max().sort_values("average_rating",ascending=True).head()
            testmin=testmin[["item_id","title"]]
            final.append(testmin)
    else:
        raise Exception("rewrite the genre")
    return final

#Unweighteditemdata("unknown")
#Weighteditemdata(30,"Horror")



    



def specifyByUserData(users, ratings, categ):
    # user based can be classified by "age", "gender", "occupation", "zip_code"
    # can specify what we wanna analyze from categ input
    user_header = ["user_id"]
    user_header.extend(categ)
    _user = users.loc[:, user_header]
    df = pd.merge(_user, ratings, on=['user_id'])
    return df


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
    _item = items.loc[:, item_header]
    df = pd.merge(_item, ratings, on=['item_id'])
    return df

# TODO - group zipcodes by this lib from https://www.zipcode.com.ng/2022/06/list-of-5-digit-zip-codes-united-states.html - steven
# REMEMBER GUYS, read table from html in pandas exist. no need for awesome webcrawling acrobatics


''' similarities '''
# TODO - connect the ratings ID to item
# TODO - compare user info with genres
