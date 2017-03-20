import numpy as np
import scipy.sparse as sp

# Here we are gonna check how the matrix in numpy are done
m = np.genfromtxt(fname="blah", delimiter="\t", usecols=(0, 3), filling_values=None,
                  dtype="S11,O")


# Accepts a matrix with [user_id],[item_id] elements
def _get_dimensions(train_data):
    uids = set()
    iids = set()

    for row in m:
        uids.add(row[0])
        iids.add(row[1])

    rows = len(uids) + 1
    cols = len(iids) + 1

    return rows, cols

def _build_interaction_matrix(rows, cols, data, min_rating):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, _ in data:
        if rating >= min_rating:
            mat[uid, iid] = rating

    return mat.tocoo()



rows, cols = _get_dimensions(m)




print "Rows: ", rows, "Cols: ", cols
