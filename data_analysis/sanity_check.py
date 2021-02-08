import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt

'''
The neural data here is BINNED

Sanity checks on general purpose data (not using random forest)

Neural responses are now in the form : response - baseline

We will see if on average, what electrodes exhibit a proper fold increase for the 
area that they are expected to be in

For fold increase we need to see the electrodes that perform the fold increase
for both halves of the data

We MIGHT also want to run Wilcoxon signed rank test 

--Wilcoxon signed rank test might need to be done on individual time bins to find the 
  number of statistically significant time steps

'''

# DEFINE CONSTANTS
category_mapping = {'House': 10, 'Visage': 20, 'Animal': 30, 'Scene': 40, 'Tool': 50, 'Scramble': 90}
brodmann_area_mapping = {'V1': 17, 'V2': 18, 'V3': 19, 'V4': 19, 'IT': 20,
                         'Fusiform Gyrus': 37, 'Parahippocampal Gyrus': 34}
comparison_categories_face = ['House', 'Scene', 'Tool']
comparison_categories_scenes = ['Visage', 'Animal', 'Tool']
comparison_categories_IT = ['Scramble']

categories_IT = ['House', 'Visage', 'Animal', 'Scene', 'Tool']

storage_loc = Path(
    "D:/MIT/EcogData/brainscore_img_elec_time_70hz150_mni_ba_sid_subtrbaseline/brainscore_img_elec_time_70hz150_mni_ba_sid_subtrbaseline")
neural_responses = np.load(storage_loc / "neural_responses.npy")
categories = np.load(storage_loc / 'stimgroups.npy')
brodmann_areas = np.load(storage_loc / "brodmann_areas.npy")
mni_coords = np.load(storage_loc / "mni_coordinates.npy")
subject_ids = np.load(storage_loc / "subject_ids.npy")

THRESHOLD = 1.5
TIME_BINS = 32
USE_BRODMANN = False

# bounding box for FFA left hemisphere:
ffa_x_left = [-44, -38]
ffa_y_left = [-61, -50]
ffa_z_left = [-24, -15]

# bounding box for FFA Right Hemisphere:
ffa_x_right = [36, 43]
ffa_y_right = [-55, -49]
ffa_z_right = [-25, -13]

# bounding box for PPA left hemisphere:
ppa_x_left = [-31, -22]
ppa_y_left = [-55, -49]
ppa_z_left = [-12, -6]

# bounding box for PPA right hemisphere:
ppa_x_right = [24, 32]
ppa_y_right = [-54, -45]
ppa_z_right = [-12, -6]


def get_IT_electrodes():
    IT_electrodes = np.zeros(neural_responses.shape[1])
    IT_idx = brodmann_areas == brodmann_area_mapping['IT']
    return IT_idx.astype(int)


def get_ffa_electrodes():
    if USE_BRODMANN:
        ffa_electrodes = np.zeros(neural_responses.shape[1])
        ffa_idx = brodmann_areas == brodmann_area_mapping['Fusiform Gyrus']
        return ffa_idx.astype(int)

    ffa_electrodes = np.zeros(neural_responses.shape[1])
    for idx, electrode in enumerate(mni_coords):
        x_coord = electrode[0]
        y_coord = electrode[1]
        z_coord = electrode[2]

        if (((x_coord > ffa_x_left[0]) and (x_coord < ffa_x_left[1]) and
             (y_coord > ffa_y_left[0]) and (y_coord < ffa_y_left[1]) and
             (z_coord > ffa_z_left[0]) and (z_coord < ffa_z_left[1])) or
                ((x_coord > ffa_x_right[0]) and (x_coord < ffa_x_right[1]) and
                 (y_coord > ffa_y_right[0]) and (y_coord < ffa_y_right[1]) and
                 (z_coord > ffa_z_right[0]) and (z_coord < ffa_z_right[1]))):
            ffa_electrodes[idx] = 1

    return ffa_electrodes


def get_ppa_electrodes():
    if USE_BRODMANN:
        ppa_electrodes = np.zeros(neural_responses.shape[1])
        ppa_idx = brodmann_areas == brodmann_area_mapping['Parahippocampal Gyrus']
        return ppa_idx.astype(int)


    ppa_electrodes = np.zeros(neural_responses.shape[1])
    for idx, electrode in enumerate(mni_coords):
        x_coord = electrode[0]
        y_coord = electrode[1]
        z_coord = electrode[2]

        if (((x_coord > ppa_x_left[0]) and (x_coord < ppa_x_left[1]) and
             (y_coord > ppa_y_left[0]) and (y_coord < ppa_y_left[1]) and
             (z_coord > ppa_z_left[0]) and (z_coord < ppa_z_left[1])) or
                ((x_coord > ppa_x_right[0]) and (x_coord < ppa_x_right[1]) and
                 (y_coord > ppa_y_right[0]) and (y_coord < ppa_y_right[1]) and
                 (z_coord > ppa_z_right[0]) and (z_coord < ppa_z_right[1]))):
            ppa_electrodes[idx] = 1

    return ppa_electrodes


def _get_mean_activation(electrodes, category, half):
    category_num = category_mapping[category]
    idx = categories == category_num
    num_images = sum(idx)
    activations = neural_responses[:, electrodes.astype(bool), :]
    activations = activations[idx]

    if half == 'First':
        activations = activations[:math.floor(num_images / 2)]
    elif half == 'Second':
        activations = activations[math.floor(num_images / 2):]

    return np.mean(activations, axis=0)


# averages the activations for the ffa electrodes for face images and
# also the comparison images
def get_activations_IT():
    # get electrodes`
    IT_electrodes = get_IT_electrodes()

    IT_activations_one = np.zeros((len(categories_IT), sum(IT_electrodes), TIME_BINS))
    IT_activations_two = np.zeros((len(categories_IT), sum(IT_electrodes), TIME_BINS))

    # get the average activations for all other categories:
    for idx, category in enumerate(categories_IT):
        IT_activations_one[idx, :, :] = _get_mean_activation(IT_electrodes, category, 'First')
        IT_activations_two[idx, :, :] = _get_mean_activation(IT_electrodes, category, 'Second')

    IT_activations_one = np.mean(IT_activations_one, axis=0)
    IT_activations_two = np.mean(IT_activations_two, axis=0)

    # initialize the comparison categories Array Size: (3, 10, 32)
    other_activations_one = np.zeros((len(comparison_categories_IT), np.shape(IT_activations_one)[0],
                                      np.shape(IT_activations_one)[1]))

    other_activations_two = np.zeros((len(comparison_categories_IT), np.shape(IT_activations_one)[0],
                                      np.shape(IT_activations_one)[1]))

    # get the average activations for all other categories:
    for idx, category in enumerate(comparison_categories_IT):
        other_activations_one[idx, :, :] = _get_mean_activation(IT_electrodes, category, 'First')
        other_activations_two[idx, :, :] = _get_mean_activation(IT_electrodes, category, 'Second')

    other_activations_one = np.mean(other_activations_one, axis=0)
    other_activations_two = np.mean(other_activations_two, axis=0)

    return IT_activations_one, IT_activations_two, other_activations_one, other_activations_two


# averages the activations for the ffa electrodes for face images and
# also the comparison images
def get_activations_ffa():
    # get electrodes
    ffa_electrodes = get_ffa_electrodes()

    # get average activations for those 10 electrode. Array size: (10, 32)
    face_activations_one = _get_mean_activation(ffa_electrodes, 'Visage', 'First')
    face_activations_two = _get_mean_activation(ffa_electrodes, 'Visage', 'Second')

    # initialize the comparison categories Array Size: (3, 10, 32)
    other_activations_one = np.zeros((len(comparison_categories_face), np.shape(face_activations_one)[0],
                                      np.shape(face_activations_one)[1]))

    other_activations_two = np.zeros((len(comparison_categories_face), np.shape(face_activations_one)[0],
                                      np.shape(face_activations_one)[1]))

    # get the average activations for all other categories:
    for idx, category in enumerate(comparison_categories_face):
        other_activations_one[idx, :, :] = _get_mean_activation(ffa_electrodes, category, 'First')
        other_activations_two[idx, :, :] = _get_mean_activation(ffa_electrodes, category, 'Second')

    other_activations_one = np.mean(other_activations_one, axis=0)
    other_activations_two = np.mean(other_activations_two, axis=0)

    return face_activations_one, face_activations_two, other_activations_one, other_activations_two


# averages the activations for the ffa electrodes for face images and
# also the comparison images
def get_activations_ppa():
    # get electrodes
    ppa_electrodes = get_ppa_electrodes()

    # get average activations for those 10 electrode. Array size: (10, 32)
    scene_activations_one = _get_mean_activation(ppa_electrodes, 'Scene', 'First')
    scene_activations_two = _get_mean_activation(ppa_electrodes, 'Scene', 'Second')

    # initialize the comparison categories Array Size: (3, 10, 32)
    other_activations_one = np.zeros((len(comparison_categories_scenes), np.shape(scene_activations_one)[0],
                                      np.shape(scene_activations_one)[1]))

    other_activations_two = np.zeros((len(comparison_categories_scenes), np.shape(scene_activations_one)[0],
                                      np.shape(scene_activations_one)[1]))

    # get the average activations for all other categories:
    for idx, category in enumerate(comparison_categories_scenes):
        other_activations_one[idx, :, :] = _get_mean_activation(ppa_electrodes, category, 'First')
        other_activations_two[idx, :, :] = _get_mean_activation(ppa_electrodes, category, 'Second')

    other_activations_one = np.mean(other_activations_one, axis=0)
    other_activations_two = np.mean(other_activations_two, axis=0)

    return scene_activations_one, scene_activations_two, other_activations_one, other_activations_two


# returns the set of responsive electrodes for the specified time bin
def get_responsive_electrodes(region, time_bin):
    # first get the activation for both halves of the image set:
    if region == 'ffa':
        activations_one, activations_two, other_activations_one, other_activations_two = get_activations_ffa()
    elif region == 'ppa':
        activations_one, activations_two, other_activations_one, other_activations_two = get_activations_ppa()
    elif region == 'IT':
        activations_one, activations_two, other_activations_one, other_activations_two = get_activations_IT()
    else:
        return

    # extract the average activation for a particular window, each step is 31.25
    if time_bin == '50-250':
        activations_one = np.mean(activations_one[:, 2:9], axis=1)
        activations_two = np.mean(activations_two[:, 2:9], axis=1)
        other_activations_one = np.mean(other_activations_one[:, 2:9], axis=1)
        other_activations_two = np.mean(other_activations_two[:, 2:9], axis=1)
    elif time_bin == '150-350':
        activations_one = np.mean(activations_one[:, 5:12], axis=1)
        activations_two = np.mean(activations_two[:, 5:12], axis=1)
        other_activations_one = np.mean(other_activations_one[:, 5:12], axis=1)
        other_activations_two = np.mean(other_activations_two[:, 5:12], axis=1)
    elif time_bin == '250-450':
        activations_one = np.mean(activations_one[:, 9:16], axis=1)
        activations_two = np.mean(activations_two[:, 9:16], axis=1)
        other_activations_one = np.mean(other_activations_one[:, 9:16], axis=1)
        other_activations_two = np.mean(other_activations_two[:, 9:16], axis=1)

    # discard negative values
    activations_one[activations_one < 0] = np.nan
    activations_two[activations_two < 0] = np.nan
    other_activations_one[other_activations_one < 0] = np.nan
    other_activations_two[other_activations_two < 0] = np.nan

    # for each electrode we want to divide things through and see on average for all the time bins if it had
    # a greater than threshold

    factor_increase_one = np.divide(activations_one, other_activations_one)  # (10, 32) size
    factor_increase_two = np.divide(activations_two, other_activations_two)

    top_electrodes_one = factor_increase_one > THRESHOLD
    top_electrodes_two = factor_increase_two > THRESHOLD

    top_electrodes_one = [i for i, x in enumerate(top_electrodes_one) if x]
    top_electrodes_two = [i for i, x in enumerate(top_electrodes_two) if x]

    return list(set(top_electrodes_one) & set(top_electrodes_two))


def average_responsive_ffa(time_bin):
    # get the responsive electrodes
    face_activations_one, face_activations_two, other_activations_one, other_activations_two = get_activations_ffa()
    #face_activations = np.mean([face_activations_one, face_activations_two], axis=0)
    #other_activations = np.mean([other_activations_one, other_activations_two], axis=0)
    responsive_electrodes = get_responsive_electrodes('ffa', time_bin)

    # now let us look at only the responsive electrodes
    face_activations = np.mean(face_activations_two[responsive_electrodes], axis=0)
    other_activations = np.mean(other_activations_two[responsive_electrodes], axis=0)

    # get the average of their responses to a particular category
    time_bins = np.arange(0, 1000, 31.25)

    if time_bin == '50-250':
        face_activations = face_activations[2:9]
        other_activations = other_activations[2:9]
        time_bins = time_bins[2:9]
    elif time_bin == '150-350':
        face_activations = face_activations[5:12]
        other_activations = other_activations[5:12]
        time_bins = time_bins[5:12]
    else:
        face_activations = face_activations[9:16]
        other_activations = other_activations[9:16]
        time_bins = time_bins[9:16]

    # get the average increase
    face_activations = np.mean(face_activations)
    other_activations = np.mean(other_activations)
    factor_increase = np.divide(face_activations, other_activations)
    return factor_increase


def average_responsive_ppa(time_bin):
    # get the responsive electrodes
    scene_activations_one, scene_activations_two, other_activations_one, other_activations_two = get_activations_ppa()
    #scene_activations = np.mean([scene_activations_one, scene_activations_two], axis=0)
    #other_activations = np.mean([other_activations_one, other_activations_two], axis=0)
    responsive_electrodes = get_responsive_electrodes('ppa', time_bin)

    # now let us look at only the responsive electrodes
    scene_activations = np.mean(scene_activations_two[responsive_electrodes], axis=0)
    other_activations = np.mean(other_activations_two[responsive_electrodes], axis=0)

    # get the average of their responses to a particular category
    time_bins = np.arange(0, 1000, 31.25)

    if time_bin == '50-250':
        scene_activations = scene_activations[2:9]
        other_activations = other_activations[2:9]
        time_bins = time_bins[2:9]
    elif time_bin == '150-350':
        scene_activations = scene_activations[5:12]
        other_activations = other_activations[5:12]
        time_bins = time_bins[5:12]
    else:
        scene_activations = scene_activations[9:16]
        other_activations = other_activations[9:16]
        time_bins = time_bins[9:16]

    # get the average increase
    scene_activations = np.mean(scene_activations)
    other_activations = np.mean(other_activations)
    factor_increase = np.divide(scene_activations, other_activations)
    return factor_increase


def average_responsive_IT(time_bin):
    # get the responsive electrodes
    IT_activations_one, IT_activations_two, other_activations_one, other_activations_two = get_activations_IT()

    # IT_activations = np.mean([IT_activations_one, IT_activations_two], axis=0)
    # other_activations = np.mean([other_activations_one, other_activations_two], axis=0)

    responsive_electrodes = get_responsive_electrodes('IT', time_bin)

    # now let us look at only the responsive electrodes
    IT_activations = np.mean(IT_activations_two[responsive_electrodes], axis=0)
    other_activations = np.mean(other_activations_two[responsive_electrodes], axis=0)

    # get the average of their responses to a particular category
    time_bins = np.arange(0, 1000, 31.25)

    if time_bin == '50-250':
        IT_activations = IT_activations[2:9]
        other_activations = other_activations[2:9]
        time_bins = time_bins[2:9]
    elif time_bin == '150-350':
        IT_activations = IT_activations[5:12]
        other_activations = other_activations[5:12]
        time_bins = time_bins[5:12]
    else:
        IT_activations = IT_activations[9:16]
        other_activations = other_activations[9:16]
        time_bins = time_bins[9:16]

    # get the average increase
    IT_activations = np.mean(IT_activations)
    other_activations = np.mean(other_activations)
    factor_increase = np.divide(IT_activations, other_activations)
    return factor_increase


def plot_average_response():
    time_bin1_IT = average_responsive_IT('50-250')
    time_bin2_IT = average_responsive_IT('150-350')
    time_bin3_IT = average_responsive_IT('250-450')
    times = ['50-250', '150-350', '250-450']
    average_increases = [time_bin1_IT, time_bin2_IT, time_bin3_IT]

    fig = plt.figure()
    x_pos = [i for i, _ in enumerate(times)]
    plt.style.use('ggplot')
    plt.bar(x_pos, average_increases, color='green')

    plt.xlabel("Times")
    plt.ylabel("Fold Increase")
    plt.title("IT")
    plt.xticks(x_pos, times)

    time_bin1_ppa = average_responsive_ppa('50-250')
    time_bin2_ppa = average_responsive_ppa('150-350')
    time_bin3_ppa = average_responsive_ppa('250-450')
    average_increases = [time_bin1_ppa, time_bin2_ppa, time_bin3_ppa]
    fig = plt.figure()
    x_pos = [i for i, _ in enumerate(times)]
    plt.style.use('ggplot')
    plt.bar(x_pos, average_increases, color='green')

    plt.xlabel("Times")
    plt.ylabel("Fold Increase")
    plt.title("PPA")
    plt.xticks(x_pos, times)

    # FFA, average fold increase over comparison categories
    # time bin 1 (approximate):
    time_bin1_ffa = average_responsive_ffa('50-250')
    time_bin2_ffa = average_responsive_ffa('150-350')
    time_bin3_ffa = average_responsive_ffa('250-450')
    average_increases = [time_bin1_ffa, time_bin2_ffa, time_bin3_ffa]
    fig = plt.figure()
    x_pos = [i for i, _ in enumerate(times)]
    plt.style.use('ggplot')
    plt.bar(x_pos, average_increases, color='green')

    plt.xlabel("Times")
    plt.ylabel("Fold Increase")
    plt.title("FFA")
    plt.xticks(x_pos, times)
    print('here')

def list_responsive_electrodes():

    list1 = get_responsive_electrodes('ffa', '50-250')
    list2 = get_responsive_electrodes('ffa', '150-350')
    list3 = get_responsive_electrodes('ffa', '250-450')
    unique_subs = get_unique_subjects(list1)
    unique_subs2 = get_unique_subjects(list2)
    unique_subs3 = get_unique_subjects(list3)

    #print(list1, list2,  list3)
    print("FFA", unique_subs, unique_subs2, unique_subs3)


    list1 = get_responsive_electrodes('ppa', '50-250')
    list2 = get_responsive_electrodes('ppa', '150-350')
    list3 = get_responsive_electrodes('ppa', '250-450')
    unique_subs = get_unique_subjects(list1)
    unique_subs2 = get_unique_subjects(list2)
    unique_subs3 = get_unique_subjects(list3)

    #print(list1, list2, list3)
    print("PPA", unique_subs, unique_subs2, unique_subs3)

    list1 = get_responsive_electrodes('IT', '50-250')
    list2 = get_responsive_electrodes('IT', '150-350')
    list3 = get_responsive_electrodes('IT', '250-450')
    unique_subs = get_unique_subjects(list1)
    unique_subs2 = get_unique_subjects(list2)
    unique_subs3 = get_unique_subjects(list3)

    #print(list1, list2, list3)
    print("IT", unique_subs, unique_subs2, unique_subs3)

def get_unique_subjects(electrodes):
    sub_ids = subject_ids[electrodes]
    unique_subs = set(sub_ids)
    return unique_subs

if __name__ == '__main__':
    #plot_average_response()
    #list_responsive_electrodes()
    list_responsive_electrodes()








