import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
from pathlib import Path

loc_Ecog =  Path("C:/Users/hsuen/Desktop/bigData/brainscore_img_elec_time_70hz150")

loc_scores = "C:/Users/hsuen/.result_caching/"
model = 'alexnet'
activations_extractor_loc = "model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored/identifier=alexnet,stimuli_identifier=aru.Kuzovkin2018-1trials.pkl"

def get_activations_alexnet():
    file = open(loc_scores + activations_extractor_loc, 'rb')
    x = pickle.load(file)
    return x['data']

def get_electrode_activations():
    neural_response = np.load(loc_Ecog / "neural_responses.npy")


    # extract out only the relevant electrodes and the 50-150ms time bin
    neural_response = neural_response[:, [1094, 847], 2:4]
    neural_response = np.mean(neural_response, axis=2)
    return neural_response


def construct_RDM(activations):
    num_images = len(activations)
    RDM = np.zeros((num_images, num_images))

    for x in range(num_images):
        for y in range(num_images):
            # get the pearson correlation
            correl = 1 - (pearsonr(activations[x][:], activations[y][:]))[0]
            RDM[x][y] = correl
    return RDM


if __name__ == "__main__":
    elec_activ = get_electrode_activations()
   # alexnet_activ = get_activations_alexnet()

    RDM_elec = construct_RDM(elec_activ)
   # RDM_alex = construct_RDM(alexnet_activ)
    print('what in the world')