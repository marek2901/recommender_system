import os

from data_path_helper import datasets_path

import findspark
findspark.init()
from pyspark.context import SparkContext
sc = SparkContext.getOrCreate()

small_ratings_file = os.path.join(datasets_path(), 'ml-latest-small', 'ratings.csv')
small_movies_file = os.path.join(datasets_path(), 'ml-latest-small', 'movies.csv')

def small_ratings_data():
    print 'returning small raitings'

    small_ratings_raw_data = sc.textFile(small_ratings_file)
    small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

    small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
                                               .map(lambda line: line.split(","))\
                                               .map(lambda tokens: (tokens[0],tokens[1],tokens[2]))\
                                               .cache()
    return small_ratings_data

def small_movies_data():
    small_movies_raw_data = sc.textFile(small_movies_file)
    small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

    small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
                                             .map(lambda line: line.split(","))\
                                             .map(lambda tokens: (tokens[0],tokens[1]))\
                                             .cache()
    return small_movies_data

def prepare_test_data(data_to_prepare):
    training_RDD, validation_RDD, test_RDD = data_to_prepare.randomSplit([6, 2, 2], seed=0L)
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    return {
        'training' : training_RDD,
        'validation' : validation_RDD,
        'test' : test_RDD,
        'valid_for_pred' : validation_for_predict_RDD,
        'test_for_predict' : test_for_predict_RDD
    }

from pyspark.mllib.recommendation import ALS
import math

def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

def train(dataset):
    prepd_data = prepare_test_data(dataset)
    seed = 5L
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4, 8, 12]
    errors = [0, 0, 0]
    err = 0
    tolerance = 0.02

    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    for rank in ranks:
        model = ALS.train(prepd_data['training'], rank, seed=seed, iterations=iterations,
                          lambda_=regularization_parameter)
        predictions = model.predictAll(prepd_data['valid_for_pred']).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = prepd_data['validation'].map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err += 1
        print 'For rank %s the RMSE is %s' % (rank, error)
        if error < min_error:
            min_error = error
            best_rank = rank
    print 'The best model was trained with rank %s' % best_rank

    model = ALS.train(prepd_data['training'], best_rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(prepd_data['test_for_predict']).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = prepd_data['test'].map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

    print 'For testing data the RMSE is %s' % (error)


def complete_data_model_pred():
    seed = 5L
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4, 8, 12]
    errors = [0, 0, 0]
    err = 0
    tolerance = 0.02

    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    # Load the complete dataset file
    complete_ratings_file = os.path.join(datasets_path(), 'ml-latest', 'ratings.csv')
    complete_ratings_raw_data = sc.textFile(complete_ratings_file)
    complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

    # Parse
    complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
                                                     .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

    print "There are %s recommendations in the complete dataset" % (complete_ratings_data.count())

    training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0L)

    complete_model = ALS.train(training_RDD, best_rank, seed=seed,
                               iterations=iterations, lambda_=regularization_parameter)

    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

    print 'For testing data the RMSE is %s' % (error)

    complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
    complete_movies_raw_data = sc.textFile(complete_movies_file)
    complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

    # Parse
    complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
                                                   .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()

    complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))

    print "There are %s movies in the complete dataset" % (complete_movies_titles.count())

    movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
    movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
    movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


if __name__ == "__main__":
    # Print example data when loaded as main module
    # just for test purpoese
     train(small_ratings_data())

    # complete_data_model_pred()
