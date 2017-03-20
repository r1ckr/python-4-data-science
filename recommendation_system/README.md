# recommender_system_challenge
Recommender System Challenge by @Sirajology on [Youtube](https://youtu.be/9gBC9R-msAk).

## Overview

This is the code for the Recommender System challenge for 'Learn Python for Data Science #3' by @Sirajology on [YouTube](https://youtu.be/9gBC9R-msAk). The code uses the [lightfm](https://github.com/lyst/lightfm) recommender system library to train a hybrid content-based + collaborative algorithm that uses the WARP loss function on the [movielens](http://grouplens.org/datasets/movielens/) dataset. The movielens dataset contains movies and ratings from over 1700 users. Once trained, our script prints out recommended movies for whatever users from the dataset that we choose to terminal.

## Dependencies

* numpy (http://www.numpy.org/)
* scipy (https://www.scipy.org/)
* lightfm (https://github.com/lyst/lightfm)

Install missing dependencies using [pip](https://pip.pypa.io/en/stable/installing/)

```
pip install -r requirements.txt
```

## Running it
To run this, first extract the data files:
cd data/books
unzip BX-CSV-Dump.zip