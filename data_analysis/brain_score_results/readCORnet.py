import pickle
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import pickle

scores_loc = '/Users/hermessuen/Documents/KanwisherLab/brainscore_results/'
layer_scores_loc = 'model_tools.brain_transformation.neural.LayerScores._call/'
layer_selection_loc = 'model_tools.brain_transformation.neural.LayerSelection._call/'
model_score_loc = 'brainscore.score_model/'

CORnet_commitment_loc = 'candidate_models.model_commitments.cornets.CORnetCommitment.look_at_cached/'



CORNET_MODELS = ['CORnet-S', 'CORnet-R', 'CORnet-Z', 'CORnet-R2']

def read_model_scores():
    scores = []
    for model in CORNET_MODELS:
        file_to_read = 'model_identifier=' + model + ",benchmark_identifier=aru.Kuzovkin2018-pls.pkl"
        with open(scores_loc+model_score_loc + file_to_read, "rb") as f:
            score = pickle.load(f)
            scores.append(float(score['data'].raw[0].values))

    return CORNET_MODELS, scores

def read_commitment_scores():
    for model in CORNET_MODELS:
        file_to_read = 'model_identifier=' + model + ",stimuli_identifier=aru.Kuzovkin2018.pkl"
        with open(scores_loc + CORnet_commitment_loc + file_to_read, "rb") as f:
            score = pickle.load(f)
            print('Fucking here')

if __name__ == '__main__':
    #read_model_scores()
    read_commitment_scores()