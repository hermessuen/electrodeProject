
# set up the environment variables
import os



import functools
import logging

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

import brainio_collection
from brainio_collection.fetch import BotoFetcher
from brainscore.benchmarks._neural_common import NeuralBenchmark
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.utils import LazyLoad
import tensorflow as tf
import keras
#import os

_logger = logging.getLogger(__name__)


from brainscore import score_model

models = ['alexnet']
#
# models = ['squeezenet1_0', 'squeezenet1_1',
#           'xception', 'densenet-121', 'densenet-169', 'densenet-201', 'inception_v1',
#           'inception_v2', 'inception_v3', 'inception_v4', 'inception_resnet_v2',
#           'resnet-18', 'resnet-34', 'resnet-50-pytorch', 'resnet-50_v1',
#           'resnet-101_v1', 'resnet-152_v1', 'resnet-50_v2', 'resnet-101_v2',
#           'resnet-152_v2', 'resnet-50-robust', 'vgg-16', 'vgg-19', 'vggface',
#           'bagnet9', 'bagnet17', 'bagnet33', 'nasnet_mobile', 'nasnet_large',
#           'pnasnet_large', 'mobilenet_v1_1.0_224', 'mobilenet_v1_1.0_192', 'mobilenet_v1_1.0_160',
#           'mobilenet_v1_1.0_128']




# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from candidate_models.model_commitments import brain_translated_pool

for x in models:
    model = brain_translated_pool[x]
    try:
        score = score_model(model_identifier=x, model=model, benchmark_identifier='aru.Kuzovkin2018-pls')

    except Exception as e:
        print('SOMETHING WENT WRONG. Tried to score {0} and failed. Here is the Exception'.format(x))
        print(e)


print('Done scoring')

