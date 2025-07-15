#!/usr/bin/python
#
# This file is part of PyRQA.
# Copyright 2015 Tobias Rawald, Mike Sips.

"""
Distance metrics.
"""

import math

import numpy as np

from pyrqa.abstract_classes import AbstractMetric


class TaxicabMetric(AbstractMetric):
    """
    Taxicab metric (L1)
    """
    name = 'taxicab_metric'

    @classmethod
    def get_distance_time_series(cls, time_series_x, time_series_y, embedding_dimension, time_delay, index_x, index_y):
        """ See AbstractMetric """
        distance = 0
        for idx in np.arange(embedding_dimension):
            temp_x = index_x + (idx * time_delay)
            temp_y = index_y + (idx * time_delay)

            distance += math.fabs(time_series_x[temp_x] - time_series_y[temp_y])

        return distance

    @classmethod
    def get_distance_vectors(cls, vectors_x, vectors_y, embedding_dimension, index_x, index_y):
        """ See AbstractMetric """
        distance = 0
        for idx in np.arange(embedding_dimension):
            temp_x = index_x * embedding_dimension + idx
            temp_y = index_y * embedding_dimension + idx

            distance += math.fabs(vectors_x[temp_x] - vectors_y[temp_y])

        return distance


class EuclideanMetric(AbstractMetric):
    """
    Euclidean metric (L2)
    """
    name = 'euclidean_metric'

    @classmethod
    def get_distance_time_series(cls, time_series_x, time_series_y, embedding_dimension, time_delay, index_x, index_y):
        """ See AbstractMetric """
        distance = 0
        for idx in np.arange(embedding_dimension):
            temp_x = index_x + (idx * time_delay)
            temp_y = index_y + (idx * time_delay)

            distance += math.pow(time_series_x[temp_x] - time_series_y[temp_y], 2)

        return math.sqrt(distance)

    @classmethod
    def get_distance_vectors(cls, vectors_x, vectors_y, embedding_dimension, index_x, index_y):
        """ See AbstractMetric """
        distance = 0
        for idx in np.arange(embedding_dimension):
            temp_x = index_x * embedding_dimension + idx
            temp_y = index_y * embedding_dimension + idx

            distance += math.pow(vectors_x[temp_x] - vectors_y[temp_y], 2)

        return math.sqrt(distance)


class MaximumMetric(AbstractMetric):
    """
    Maximum metric (L_inf)
    """
    name = 'maximum_metric'

    @classmethod
    def get_distance_time_series(cls, time_series_x, time_series_y, embedding_dimension, time_delay, index_x, index_y):
        """ See AbstractMetric """
        distance = np.finfo(np.float32).min
        for index in np.arange(embedding_dimension):
            temp_x = index_x + (index * time_delay)
            temp_y = index_y + (index * time_delay)

            value = math.fabs(time_series_x[temp_x] - time_series_y[temp_y])
            if value > distance:
                distance = value

        return distance

    @classmethod
    def get_distance_vectors(cls, vectors_x, vectors_y, embedding_dimension, index_x, index_y):
        """ See AbstractMetric """
        distance = np.finfo(np.float32).min
        for idx in np.arange(embedding_dimension):
            temp_x = index_x * embedding_dimension + idx
            temp_y = index_y * embedding_dimension + idx

            value = math.fabs(vectors_x[temp_x] - vectors_y[temp_y])
            if value > distance:
                distance = value

        return distance

class CosineMetric(AbstractMetric):
    """
    Cosine distance: 1 - (x · y) / (‖x‖‖y‖)
    """
    name = 'cosine_metric'

    @classmethod
    def get_distance_time_series(cls,
                                 time_series_x, time_series_y,
                                 embedding_dimension, time_delay,
                                 index_x, index_y):
        # build the two embedded vectors
        x_vec = np.array([
            time_series_x[index_x + i * time_delay]
            for i in range(embedding_dimension)
        ], dtype=float)
        y_vec = np.array([
            time_series_y[index_y + i * time_delay]
            for i in range(embedding_dimension)
        ], dtype=float)

        # dot product and norms
        dot = np.dot(x_vec, y_vec)
        norm_x = np.linalg.norm(x_vec)
        norm_y = np.linalg.norm(y_vec)
        # guard against zero‐vector
        if norm_x == 0 or norm_y == 0:
            return 1.0
        # cosine distance
        return 1.0 - (dot / (norm_x * norm_y))

    @classmethod
    def get_distance_vectors(cls,
                             vectors_x, vectors_y,
                             embedding_dimension,
                             index_x, index_y):
        # flatten and slice out each embedded vector
        offset_x = index_x * embedding_dimension
        offset_y = index_y * embedding_dimension
        x_vec = vectors_x[offset_x:offset_x + embedding_dimension].astype(float)
        y_vec = vectors_y[offset_y:offset_y + embedding_dimension].astype(float)

        dot = np.dot(x_vec, y_vec)
        norm_x = np.linalg.norm(x_vec)
        norm_y = np.linalg.norm(y_vec)
        if norm_x == 0 or norm_y == 0:
            return 1.0
        return 1.0 - (dot / (norm_x * norm_y))
