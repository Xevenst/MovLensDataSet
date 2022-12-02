import pandas as pd

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
        lambda x: x[-4:] if type(x)==str else x)
    # the only columns that matter is just id and genres hahahaah
    movies = movies.drop(
        columns=['video_release_date', "title", "IMDb_URL"])
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
    users["age_category"] = pd.cut(users["age"], bins=[
                                   0, 10, 20, 30, 40, 50, 60, 70, 80], labels=[1, 2, 3, 4, 5, 6, 7, 8])
    return users

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
    if categ == "genres":
        item_header.extend(["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
                           "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    else:
        return 0
    _item = items.loc[:, item_header]
    df = pd.merge(_item, ratings, on=['item_id'])
    return df

# TODO - group zipcodes by this lib from https://www.zipcode.com.ng/2022/06/list-of-5-digit-zip-codes-united-states.html - steven
# REMEMBER GUYS, read table from html in pandas exist. no need for awesome webcrawling acrobatics

''' similarities '''
# TODO - connect the ratings ID to item
# TODO - compare user info with genres