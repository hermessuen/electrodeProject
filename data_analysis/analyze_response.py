import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# Constants defined here:
category_mapping = {'House': 10, 'Visage': 20, 'Animal': 30, 'Scene': 40, 'Tool': 50, 'Scramble': 90}

brodmann_area_mapping = {'V1': 17, 'V2': 18, 'V3': 19, 'V4': 19, 'IT': 20, 'Primary Visual': 5}
storage_location = Path(
    "C:/Users/hsuen/OneDrive - Mathworks/Desktop/MITLab/electrodeProject/data1/brainscore_img_elec_time_70hz150"
    "/brainscore_img_elec_time_70hz150/")


# pass in the brodmann area where you want to get the number of electrodes in
def get_num_electrodes_in_area(area):
    brodmann_areas = np.load(storage_location / "brodmann_areas.npy")
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
def calculate_mean_activation(category, region, time_bin):
    # LOAD DATA
    neural_responses = np.load(storage_location / "neural_responses.npy")
    categories = np.load(storage_location / 'stimgroups.npy')
    brodmann_areas = np.load(storage_location / "brodmann_areas.npy")

    category_num = category_mapping[category]

    # CALCULATE THE MEAN

    # extract responses for specific category
    idx = categories == category_num
    activation_mean = neural_responses[idx][:][:]  # (images, 11293, 32)

    # compute average for each electrode across all images
    activation_mean = np.mean(activation_mean, axis=0)  # (11293, 32)

    # extract responses for particular electrodes
    if region == 'All':
        # compute average for all electrodes
        activation_mean = np.mean(activation_mean, axis=0)  # (32,)
    else:
        electrode_num = brodmann_area_mapping[region]
        idx = brodmann_areas == electrode_num
        activation_mean = activation_mean[idx][:]
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
    factor_increase_noise = np.divide(category_activations, noise_activations) # (32,)



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
    ax = plt.gca()

    # create a line @ y = 1
    x1 = np.ones((32, 1))
    plt.plot(x1, 'r--')
    ax.legend(['Other', 'Noise'])

    plt.show()





if __name__ == '__main__':
    category_impact('Scene', 'IT')