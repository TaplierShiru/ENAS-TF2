from typing import Tuple
import tensorflow as tf

import numpy as np


def apply_drop_path(x_in, prob):
    batch_size = tf.shape(x_in)[0]
    noise_shape = [batch_size, 1, 1, 1]
    random_tensor = prob
    random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
    binary_tensor = tf.floor(random_tensor)
    return tf.div(x_in, prob) * binary_tensor


def generate_random_nas_arc(num_cells: int, num_branches=5) -> Tuple[list, list]:
    """
    Returns normal arc and reduce arc as two lists

    """
    normal_arc = []
    reduce_arc = []
    generate_single_cell = lambda layer_id: [
        #          x id                         x op
        np.random.randint(0, layer_id+2), np.random.randint(0, num_branches),
        #          y id                         y op
        np.random.randint(0, layer_id+2), np.random.randint(0, num_branches),
    ]
    for layer_id in range(num_cells):
        # X
        normal_arc += generate_single_cell(layer_id)
        reduce_arc += generate_single_cell(layer_id)
    return normal_arc, reduce_arc


class NASCellType:
    X_TYPE = 'x'
    Y_TYPE = 'y'

class NASCellTypeV1:
    SIZE = 2
    X_TYPE = 0
    Y_TYPE = 1

class NASModelType:
    NAS_MODE = 'nas-mode'
    FIXED_MODE = 'fixed-mode'

class NASLayerType:
    REDUCTION_LAYER = 'reduction-layer'
    NORMAL_LAYER = 'normal-layer'
    AUX_LAYER = 'aug-layer'


class NASLayerTypeV1:
    SIZE = 3
    REDUCTION_LAYER = 0
    NORMAL_LAYER = 1
    AUX_LAYER = 2


class ClipGradsMode:
    NORM = 'norm'
    GLOBAL = 'global'

    