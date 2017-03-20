import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format it
data = fetch_movielens(min_rating=4.0)

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
    # We can do this here because in Python is valid to return more than 1 element

    for user_id in user_ids:

        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # tocsr = to Compressed Sparse Row Format. Here we are asking for the 'item_labels' element of the dictionary
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