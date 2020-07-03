if __name__ == '__main__':
    import numpy as np

    # first import the data
    storage_location = "C:\\Users\\hsuen\\Desktop\\MITLab\\electrodeProject\\data1\\brainscore_img_elec_time_70hz150" \
                       "\\brainscore_img_elec_time_70hz150\\"

    neural_responses = np.load(storage_location + 'neural_responses.npy')
    categories = np.load(storage_location + 'stimgroups.npy')

    category_mapping = {10: 'House', 20: 'Visage', 30: 'Animal', 40: 'Scene', 50: 'Tool', 90: 'Scramble'}
    category_keys = list(category_mapping)

    num_categories = len(category_mapping)

    # pre-allocate data
    meta_data = np.zeros((num_categories, np.shape(neural_responses)[2]))

    # loop through all of the electrodes
    for x in range(num_categories):
        idx = categories == category_keys[x]
        current_responses = neural_responses[idx][:][:]  # (images, 11293, 32)
        # compute average for each electrode across all images
        current_responses = np.mean(current_responses, axis=0)  # (11293, 32)

        # compute average for all electrodes
        current_responses = np.mean(current_responses, axis=0)  # (32,)
        meta_data[x][:] = current_responses

    # perform some data analysis to determine the xfold increase across types of images
    print('For Time Bin 1')
    for x in range(num_categories):
        factor_increase = meta_data[x][10] / meta_data[-1][10]
        print('The factor increase from noise to {0} is {1}'.format(category_mapping[category_keys[x]], factor_increase))

