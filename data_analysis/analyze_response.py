import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import math
import statsmodels.stats.multitest as multitest

# Constants defined here:
category_mapping = {'House': 10, 'Visage': 20, 'Animal': 30, 'Scene': 40, 'Tool': 50, 'Scramble': 90}

comparison_categories_scenes = ['Visage', 'Animal', 'Tool']
comparison_categories_face = ['House', 'Scene', 'Tool']

# 37 and 34 deal with Faces and Scenes respectively in the brain
brodmann_area_mapping = {'V1': 17, 'V2': 18, 'V3': 19, 'V4': 19, 'IT': 20,
                         'Fusiform Gyrus': 37, 'Parahippocampal Gyrus': 34}
storage_location = Path(
    "C:/Users/hsuen/Desktop/bigData/brainscore_img_elec_time_70hz150/")

neural_responses = np.load(storage_location / "neural_responses.npy")
categories = np.load(storage_location / 'stimgroups.npy')
brodmann_areas = np.load(storage_location / "brodmann_areas.npy")

# normalize the activations
# neural_responses = (neural_responses - np.min(neural_responses)) / (np.max(neural_responses) - np.min(neural_responses))
threshold = 1.5

# pass in the brodmann area where you want to get the number of electrodes in
def get_num_electrodes_in_area(area):
    brodmann_areas = np.load(storage_location / "brodmann_areas.npy")
    area = brodmann_area_mapping[area]
    query_area = brodmann_areas == area
    return np.sum(query_area)


def get_brodmann_areas():
    brodmann_areas = np.load(storage_location / "brodmann_areas.npy")
    brodmann_areas = set(brodmann_areas)  # gets the unique brodmann areas that are in the electrodes
    num_unique = len(brodmann_areas)
    return brodmann_areas, num_unique


# categories are: House, Visage, Animal, Scene, Tool, Scramble
# regions: V1, V2, V3, V4, IT, All
# time bin : 1-32, All
def calculate_mean_activation(category, region, time_bin, return_all_electrodes, which_half=None):

    if category == 'All':
        noise_idx = category_mapping['Scramble']
        noise_idx = categories == noise_idx

        activation_mean = neural_responses[~noise_idx][:][:]
        num_images = np.shape(activation_mean)[0]
        if which_half == 'First':
            activation_mean = activation_mean[:math.floor(num_images / 2)]
        elif which_half == 'Second':
            activation_mean = activation_mean[math.floor(num_images / 2):]
        activation_mean = np.mean(activation_mean, axis=0)  # (11293, 32)
    else:
        category_num = category_mapping[category]

        # CALCULATE THE MEAN

        # extract responses for specific category
        idx = categories == category_num
        activation_mean = neural_responses[idx][:][:]  # (images, 11293, 32)
        num_images = np.shape(activation_mean)[0]
        if which_half == 'First':
            activation_mean = activation_mean[:math.floor(num_images / 2)]
        elif which_half == 'Second':
            activation_mean = activation_mean[math.floor(num_images / 2):]


        # compute average for each electrode across all images
        activation_mean = np.mean(activation_mean, axis=0)  # (11293, 32)

    # extract responses for particular electrodes
    if region == 'All':
        # compute average for all electrodes
        if return_all_electrodes:
            return activation_mean

        activation_mean = np.mean(activation_mean, axis=0)  # (32,)
    else:
        electrode_num = brodmann_area_mapping[region]
        idx = brodmann_areas == electrode_num
        activation_mean = activation_mean[idx][:]

        if return_all_electrodes:
            return activation_mean

        activation_mean = np.mean(activation_mean, axis=0)

    if time_bin == 'All':
        return activation_mean
    else:
        return activation_mean[time_bin]




# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
# performs the Wilcoxon signed rank test on the differences of two samples
# Attempts to reject the null hypothesis that the difference between the two samples throughout time
# is zero. Thus, if the percent is less than 5%, we can reject it with 95% confidence

# we also correct the p-values using a multiple tests method:
# https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

def perform_sign_test(category, region):
    top_electrodes = double_screen_electrodes(region, cutoff=threshold)
    category_activations = calculate_mean_activation(category, region, 'All', return_all_electrodes=True)
    category_activations = category_activations[top_electrodes]

    if category == 'Visage':
        comparison_categories = comparison_categories_face
    elif category == 'Scene':
        comparison_categories = comparison_categories_scenes
    elif category == 'All':
        comparison_categories = ['Scramble']

    activations_other = np.zeros((len(comparison_categories), len(category_activations), 32))

    for idx, category in enumerate(comparison_categories):
        other_act = calculate_mean_activation(category, region, 'All', return_all_electrodes=True)
        activations_other[idx] = other_act[top_electrodes]

    activations_other = np.mean(activations_other, axis=0)
    p_values = []
    for idx in range(32):
        t_statistic, p_value = wilcoxon(category_activations[:, idx], activations_other[:, idx])
        p_values.append(p_value)

    # adjust the p-values here
    reject, pvals_corrected, none, none = multitest.multipletests(p_values, alpha=0.5, method='hs')
    return pvals_corrected

def scenes_and_faces(region):
    # there is from a noise baseline

    # get the mean activation values for the category
    scene_activations = calculate_mean_activation('Scene', region, 'All', return_all_electrodes=False)
    face_activations = calculate_mean_activation('Visage', region, 'All', return_all_electrodes=False)

    # calculate the activations for all other categories
    num_categories = len(category_mapping)
    activations_other_scene = np.zeros((len(comparison_categories_scenes), 32))
    activations_other_face = np.zeros((len(comparison_categories_face), 32))

    idx = 0
    for key in category_mapping:
        if key not in comparison_categories_scenes:
            continue
        activations_other_scene[idx][:] = calculate_mean_activation(key, region, 'All', return_all_electrodes=False)
        idx += 1

    idx = 0
    for key in category_mapping:
        if key not in comparison_categories_face:
            continue
        activations_other_face[idx][:] = calculate_mean_activation(key, region, 'All', return_all_electrodes=False)
        idx += 1

    # average the activations for all other categories
    activations_other_scene = np.mean(activations_other_scene, axis=0)
    activations_other_face = np.mean(activations_other_scene, axis=0)

    # plot the two differences
    factor_increase_other_scene = np.divide(scene_activations, activations_other_scene)
    factor_increase_other_face = np.divide(face_activations, activations_other_face)

    time_bins = np.arange(0, 1000, 31.25)

    plt.figure()
    plt.plot(time_bins[:30], factor_increase_other_scene[:30])

    plt.plot(time_bins[:30], factor_increase_other_face[:30])

    plt.title('Factor Increase for Faces and Scenes in region {0}'.format(region))
    plt.xlabel('Time in ms')
    ax = plt.gca()

    ax.legend(['Scene', 'Face'])
    plt.show()


def get_top_electrodes(region, cutoff, which_half):
    # look at each electrodes response for the category the region is responsible for
    if region == 'Fusiform Gyrus':
        activations = calculate_mean_activation('Visage', region, 'All', True, which_half)
        num_electrodes = np.shape(activations)[0]
        activations_other = np.zeros((len(comparison_categories_face), num_electrodes, 32))

        idx = 0
        for key in category_mapping:
            if key not in comparison_categories_face:
                continue
            activations_other[idx][:][:] = calculate_mean_activation(key, region, 'All', True, which_half)
            idx += 1

        activations_other = np.mean(activations_other, axis=0)
    elif region == 'Parahippocampal Gyrus':
        activations = calculate_mean_activation('Scene', region, 'All', True, which_half)
        num_electrodes = np.shape(activations)[0]
        activations_other = np.zeros((len(comparison_categories_scenes), num_electrodes, 32))

        idx = 0
        for key in category_mapping:
            if key not in comparison_categories_scenes:
                continue
            activations_other[idx][:][:] = calculate_mean_activation(key, region, 'All', True, which_half)
            idx += 1

        activations_other = np.mean(activations_other, axis=0)


    elif region == 'IT':
        activations = calculate_mean_activation('All', region, 'All', True, which_half)
        activations_other = calculate_mean_activation('Scramble', region, 'All', True, which_half)

    # at this point, activations is of shape (num_electrodes, 32)
    # and activations_other is the exact same shape
    # what we want to do is to see how much of an increase there
    # is when this happens

    # now perform the factor division across all the time steps
    #factor_increase = activations / activations_other  # (num_electrodes, 32)

    factor_increase = (activations - activations_other)/ activations
    factor_increase = np.abs(factor_increase)


    # now we take the mean factor increase across time
    factor_increase = np.mean(factor_increase, axis=1)  # (num_electrodes, )

    # we can screen out the top electrodes
    top_electrodes = factor_increase > cutoff

    top_electrodes = [i for i, x in enumerate(top_electrodes) if x]

    # now we need to see which one of these will work for the other half of the images

    return top_electrodes


def double_screen_electrodes(region, cutoff):
    first_set = get_top_electrodes(region, cutoff, 'First')
    second_set = get_top_electrodes(region, cutoff, 'Second')
    return list(set(first_set) & set(second_set))


def plot_top_electrodes(region):
    # gives us the indices of the top electrodes for our given region

    top_electrodes = double_screen_electrodes(region, cutoff=threshold)

    if region == 'Fusiform Gyrus':
        activations = calculate_mean_activation('Visage', region, 'All', True)
        num_electrodes = np.shape(activations)[0]
        activations_other = np.zeros((len(comparison_categories_face), num_electrodes, 32))

        idx = 0
        for key in category_mapping:
            if key not in comparison_categories_face:
                continue
            activations_other[idx][:][:] = calculate_mean_activation(key, region, 'All', True)
            idx += 1

        activations_other = np.mean(activations_other, axis=0)

    elif region == 'Parahippocampal Gyrus':
        activations = calculate_mean_activation('Scene', region, 'All', True)
        num_electrodes = np.shape(activations)[0]
        activations_other = np.zeros((len(comparison_categories_scenes), num_electrodes, 32))

        idx = 0
        for key in category_mapping:
            if key not in comparison_categories_scenes:
                continue
            activations_other[idx][:][:] = calculate_mean_activation(key, region, 'All', True)
            idx += 1

        activations_other = np.mean(activations_other, axis=0)
    elif region == 'IT':
        activations = calculate_mean_activation('All', region, 'All', True)
        activations_other = calculate_mean_activation('Scramble', region, 'All', True)

    activations = activations[top_electrodes]
    activations_other = activations_other[top_electrodes]

    factor_increase = activations / activations_other # (num_electrodes, 32)

    # now we take the mean factor increase across time
    factor_increase = np.mean(factor_increase, axis=0)  # (32, )

    # get the average of their responses to a particular category
    time_bins = np.arange(0, 1000, 31.25)

    plt.figure()
    plt.plot(time_bins[:30], factor_increase[:30])

    plt.plot(time_bins[:30], factor_increase[:30])

    plt.title('Factor Increase for important electrodes in region {0}'.format(region))
    plt.xlabel('Time in ms')
    plt.show()


if __name__ == '__main__':
    # plot_top_electrodes('Parahippocampal Gyrus')

    ## lets get the number of electrodes in each region ##
    areas = ['IT', 'Fusiform Gyrus', 'Parahippocampal Gyrus']
    meta_data = np.zeros((3, 3))
    for idx, area in enumerate(areas):
        meta_data[0, idx] = get_num_electrodes_in_area(area)
        meta_data[1, idx] = len(double_screen_electrodes(area, threshold))


    # find the number of statistically significant time steps
    p_value_face = perform_sign_test('Visage', 'Fusiform Gyrus')
    p_value_scene = perform_sign_test('Scene', 'Parahippocampal Gyrus')
    p_value_IT = perform_sign_test('All', 'IT')

    meta_data[2, 0] = sum(p_value_IT < 0.05)
    meta_data[2, 1] = sum(p_value_face < 0.05)
    meta_data[2, 2] = sum(p_value_scene < 0.05)


    print('here')
