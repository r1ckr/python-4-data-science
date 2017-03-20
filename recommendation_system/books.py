import numpy as np
from lightfm import LightFM
import scipy.sparse as sp

# m = np.genfromtxt(fname="data/music/lastfm-dataset-1K/blah", delimiter="\t", usecols=(0, 3), filling_values=None,
#                   dtype="S11,O")


books = np.genfromtxt(fname="data/books/BX-Books.csv",
                      delimiter=";",
                      usecols=(0, 1),
                      filling_values=None,
                      dtype=[np.object, np.object],
                      skip_header=1)

ratings = np.genfromtxt(fname="data/books/BX-Book-Ratings.csv",
                        delimiter=";",
                        usecols=(0, 1, 2),
                        filling_values=None,
                        dtype=[np.object, np.object, np.object],
                        skip_header=1,
                        comments='^')

books_dic = {}
for row in books:
    books_dic[str(row[0])] = row[1]

# Replace ISBN by the book name and reformat strings
for row in ratings:
    # if row[1] in books_dic:
    #     row[1] = books_dic[row[1]]
    # Reformating data:
    row[0] = str(row[0]).replace("\"", "")
    row[1] = str(row[1]).replace("\"", "")
    row[2] = int(str(row[2]).replace("\"", ""))


# Accepts a matrix with [user_id],[item_id] elements
def _get_dimensions(train_data):
    uids = set()
    iids = set()

    for row in train_data:
        uids.add(row[0])
        iids.add(row[1])

    rows = len(uids) + 1
    cols = len(iids) + 1

    return rows, cols


rows, cols = _get_dimensions(ratings)

print rows, cols
print ratings

def _build_interaction_matrix(rows, cols, data):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating in data:
            mat[uid, iid] = rating

    return mat.tocoo()


data = _build_interaction_matrix(row,cols,ratings)
print data
# Create model | warp stands for: Weighted Approximate-Rank Pairwise
# warp uses gradient descent to make its predictions
# model = LightFM(loss='warp')
#
# # epochs are the number of runs for this training session
# model.fit(data['train'], epochs=30, num_threads=2)


exit()
# Get data from the files in data/lastfm/lastfm-dataset-1K/*
# The data must be a sp.coo_matrix of shape [n_users, n_items]
# eg
# (0, 0)	5
# (1, 0)	4
# (5, 0)	4
# (9, 0)	4
# (movie_id, user_id)	rating

# So if we have a matrix like:
# user_id - item_id - rating
# we would need to get a list from each column, then create the sparse matrix:
#
# number_users = get number of users from user_id_list
# number_items = get number of users from items_id_list
# matrix = sparse.coo_matrix((ratings,(items_id_list,user_id_list)),shape=(number_users,number_items))


# Fetch data and format it
# data = fetch_movielens(min_rating=4.0)

# print training and testing data, repr it's like toString in Java
print ("Train data")
print (repr(data['train']))
print ("Test data")
print (repr(data['test']))

# Create model | warp stands for: Weighted Approximate-Rank Pairwise
# warp uses gradient descent to make its predictions
model = LightFM(loss='warp')

# epochs are the number of runs for this training session
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape
    # We can do this here because in Python is valid to return more than 1 element :O

    for user_id in user_ids:

        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # tocsr = to Compressed Sparce Row Format. Here we are asking for the 'item_labels' element of the dictionary
        # then from that we want the *indexes* that are in the 'train' data set belonging to the user

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # scores will be a list of scores

        # Rank the item in order of most liked to least liked
        top_items = data['item_labels'][np.argsort(-scores)]
        # np.argsort will return a list of the IDs of that list ordered by the value of its elements:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        # Then because we are passing -scores it will order it from large to small


        # print out the results
        print("User %s" % user_id)
        print ("    Known positives:")

        # These 2 for loops will end both in the 3rd index thanks to the [:3]
        for x in known_positives[:3]:
            print ("        %s" % x)

        print ("    Recommended:")

        for x in top_items[:3]:
            print ("        %s" % x)


sample_recommendation(model, data, [3, 25, 450])
