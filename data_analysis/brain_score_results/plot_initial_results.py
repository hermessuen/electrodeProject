import pickle
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import pickle

import data_analysis.brain_score_results.readCORnet as readCORnet



models_with_scores = []

scores_loc = 'D:/MIT/brain_score_results/'
layer_scores_loc = 'model_tools.brain_transformation.neural.LayerScores._call/'
layer_selection_loc = 'model_tools.brain_transformation.neural.LayerSelection._call/'
layer_match_loc = 'D:/MIT/brain_score_results/modelmeta-2019.json'
model_loc = 'C:/Users/hsuen/Desktop/electrodeProject/model_list.pkl'

with open(model_loc, 'rb') as f :
    models = pickle.load(f)

def read_layer_scores(model_identifier):
    file_to_read = 'model_identifier=' + model_identifier + ',benchmark_identifier=IT,visual_degrees=8.pkl'
    file = open(scores_loc + layer_scores_loc + file_to_read, 'rb')
    score = pickle.load(file)
    return score

def get_layer_matches():
    with open(layer_match_loc) as f:
        data = json.load(f)
    return data

def read_layer_selection(model_identifier):
    file_to_read = 'model_identifier=' + model_identifier +'-pca_1000,selection_identifier=IT.pkl'
    file = open(scores_loc + layer_selection_loc + file_to_read, 'rb')
    selection_identifier = pickle.load(file)
    return selection_identifier


def get_score(model_identifier):
    score = read_layer_scores(model_identifier)
    layer_selection = read_layer_selection(model_identifier)['data']
    layers = score['data'].layer.data
    layer_selection_idx = np.where(layers == layer_selection)

    return score['data'].data[layer_selection_idx][0][0], layer_selection

def plot_scores():
    all_scores = []
    all_layers = []
    for model in models:
        try:
            score, layer_selection = get_score(model)
            models_with_scores.append(model)
            all_scores.append(score)
            all_layers.append(layer_selection)
        except:
            print('Could not score {0}'.format(model))

    # get the CORnet model scores because they aren't here
    CORnet_models, CORnet_scores = readCORnet.read_model_scores()

    for idx, model in enumerate(CORnet_models):
        models_with_scores.append(model)
        all_scores.append(CORnet_scores[idx])
        all_layers.append('No Layer')

    models_not_scored = set(models) - set(models_with_scores)

    # x_pos = [i for i, _ in enumerate(models_with_scores)]
    # plt.style.use('ggplot')
    # plt.bar(x_pos, all_scores, color='green')
    #
    # plt.xlabel("Model")
    # plt.ylabel("Score")
    # plt.title("Scores")
    # plt.xticks(x_pos, models_with_scores)
    # plt.show()

    return models_with_scores, all_scores, all_layers, models_not_scored

def export_to_csv():
    models_to_save, scores, all_layers, models_not_scored = plot_scores()
    master_data = pd.DataFrame(list(zip(models_to_save, scores, all_layers)), columns=['Models', 'Score', 'Layer'])
    master_data.to_csv(scores_loc + "model_scores.csv")


if __name__ == '__main__':
    #export_to_csv()
    x = get_layer_matches()
    print('fuck')