import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import math

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
    # LOAD DATA
    neural_responses = np.load(storage_location / "neural_responses.npy")
    categories = np.load(storage_location / 'stimgroups.npy')
    brodmann_areas = np.load(storage_location / "brodmann_areas.npy")

    if category == 'All':
        noise_idx = category_mapping['Scramble']
        noise_idx = categories == noise_idx

        activation_mean = neural_responses[not noise_idx][:][:]
        num_images = np.shape(activation_mean)[0]
        if which_half == 'First':
            activation_mean = activation_mean[:math.floor(num_images/2)]
        elif which_half == 'Second':
            activation_mean = activation_mean[math.floor(num_images/2):]
        activation_mean = np.mean(activation_mean, axis=0)  # (11293, 32)
    else:
        category_num = category_mapping[category]

        # CALCULATE THE MEAN

        # extract responses for specific category
        idx = categories == category_num
        activation_mean = neural_responses[idx][:][:]  # (images, 11293, 32)
        num_images = np.shape(activation_mean)[0]
        if which_half == 'First':
            activation_mean = activation_mean[:math.floor(num_images/2)]
        elif which_half == 'Second':
            activation_mean = activation_mean[math.floor(num_images/2):]
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


# categories are: House, Visage, Animal, Scene, Tool, Scramble
# regions: V1, V2, V3, V4, IT, All
def category_impact(category, region):
    # we measure the impact of a particular category based on how much of a factor increase
    # there is from a noise baseline

    # get the mean activation values for the category
    category_activations = calculate_mean_activation(category, region, 'All')

    # get the mean activation values for a generic noise category
    noise_activations = calculate_mean_activation('Scramble', region, 'All')

    # calculate the factor increase for every time bin
    factor_increase_noise = np.divide(category_activations, noise_activations)  # (32,)

    # calculate the activations for all other categories
    num_categories = len(category_mapping)
    activations_other = np.zeros((num_categories - 1, 32))

    idx = 0
    for key in category_mapping:
        if key == category:
            continue
        activations_other[idx][:] = calculate_mean_activation(key, region, 'All')
        idx += 1

    # average the activations for all other categories
    activations_other = np.mean(activations_other, axis=0)

    # plot the two differences
    factor_increase_other = np.divide(category_activations, activations_other)
    plt.figure()
    plt.plot(factor_increase_other)

    plt.plot(factor_increase_noise)
    plt.title('Factor Increase for {0} in the {1} region'.format(category, region))
    plt.xlabel('Time in ms')
    ax = plt.gca()

    # create a line @ y = 1
    x1 = np.ones((32, 1))
    plt.plot(x1, 'r--')
    ax.legend(['Other', 'Noise'])



# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
# performs the Wilcoxon signed rank test on the differences of two samples
# Attempts to reject the null hypothesis that the difference between the two samples throughout time
# is zero. Thus, if the percent is less than 5%, we can reject it with 95% confidence
def perform_sign_test(category, region, noise):
    category_activations = calculate_mean_activation(category, region, 'All')

    if noise == 'Other':

        if category == 'Visage':
            comparison_category = comparison_categories_face
        else:
            comparison_category = comparison_categories_scenes

        # calculate the activations for all other categories
        activations_other = np.zeros((len(comparison_category) , 32))



        idx = 0
        for key in category_mapping:
            if key not in comparison_category:
                continue
            activations_other[idx][:] = calculate_mean_activation(key, region, 'All')
            idx += 1

        # average the activations for all other categories
        activations_other = np.mean(activations_other, axis=0)
    elif noise == 'Noise':
        # get the mean activation values for a generic noise category
        activations_other = calculate_mean_activation('Scramble', region, 'All')

    t_statistic, p_value_of_t = wilcoxon(category_activations, activations_other)
    print('The p-value for  {0} & {1} in the {2} Region: '.format(category, noise, region))
    print(p_value_of_t)
    return p_value_of_t


def scenes_and_faces(region):
    # there is from a noise baseline

    # get the mean activation values for the category
    scene_activations = calculate_mean_activation('Scene', region, 'All')
    face_activations = calculate_mean_activation('Visage', region, 'All')

    # calculate the activations for all other categories
    num_categories = len(category_mapping)
    activations_other_scene = np.zeros((len(comparison_categories_scenes), 32))
    activations_other_face = np.zeros((len(comparison_categories_face), 32))


    idx = 0
    for key in category_mapping:
        if key not in comparison_categories_scenes:
            continue
        activations_other_scene[idx][:] = calculate_mean_activation(key, region, 'All')
        idx += 1

    idx= 0
    for key in category_mapping:
        if key not in comparison_categories_face:
            continue
        activations_other_face[idx][:] = calculate_mean_activation(key, region, 'All')
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
    factor_increase = activations/activations_other # (num_electrodes, 32)

    # now we take the mean factor increase across time
    factor_increase = np.mean(factor_increase, axis=1) # (num_electrodes, )

    # we can screen out the top electrodes
    top_electrodes = factor_increase > cutoff

    top_electrodes = [i for i, x in enumerate(top_electrodes) if x]

    # now we need to see which one of these will work for the other half of the images


    return top_electrodes


def double_screen_electrodes(region, cutoff):
    first_set = get_top_electrodes(region, cutoff, 'First')
    second_set = get_top_electrodes(region, cutoff, 'Second')
    return list(set(first_set) & set(second_set))



if __name__ == '__main__':
    face_top_electrodes = double_screen_electrodes('Fusiform Gyrus', cutoff=1.5)
    scene_top_electrodes = double_screen_electrodes('Parahippocampal Gyrus', cutoff=1.5)
    print('Cutoff Value is 1.5')
    print('Num good electrodes in Fusiform Gyrus is {0}'.format(len(face_top_electrodes)))
    print('Num good electrodes in Parahippocampal Gyrus is {0}'.format(len(scene_top_electrodes)))

