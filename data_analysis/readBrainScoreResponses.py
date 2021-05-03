import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# models = ['alexnet','CORnet-S', 'CORnet-R', 'CORnet-Z', 'CORnet-R2', 'squeezenet1_0', 'squeezenet1_1']

models = ['alexnet']
loc_scores = "C:/Users/hsuen/.result_caching/"


## example file names shown here ##:
benchmark_loc = "brainscore.benchmarks.NeuralBenchmark._ceiling/identifier=aru.Kuzovkin2018-pls.pkl"
activations_extractor_loc = "model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored/identifier=alexnet,stimuli_identifier=aru.Kuzovkin2018-1trials.pkl"
pca_extractor_loc = "model_tools.activations.pca.LayerPCA._pcas/identifier=alexnet-pca_1000,n_components=1000.pkl"
layer_scores = "model_tools.brain_transformation.neural.LayerScores._call/model_identifier=alexnet,benchmark_identifier=IT,visual_degrees=8.pkl"

def playground():
    file = open(loc_scores + activations_extractor_loc, 'rb')
    x = pickle.load(file)
    print('what do we have here')




def get_all_scores():
    scores = []
    for x in models:
        brain_score_model = "brainscore.score_model/model_identifier=" + x + ",benchmark_identifier=aru.Kuzovkin2018-pls.pkl"

        # load the brainscore model scores?
        file = open(loc_scores + brain_score_model, 'rb')
        score= pickle.load(file)
        scores.append(score['data'][0].values)


    x_pos = [i for i, _ in enumerate(models)]
    plt.style.use('ggplot')
    plt.bar(x_pos, scores, color='green')

    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Scores")
    plt.xticks(x_pos, models)


if __name__ == "__main__":
    get_all_scores()





















