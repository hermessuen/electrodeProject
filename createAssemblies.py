if __name__ == '__main__':
    import brainscore
    import numpy as np
    import xarray as xr

    # first import the data
    storage_location = "C:\\Users\\hsuen\\Desktop\\MITLab\\electrodeProject\\data1\\brainscore_img_elec_time_70hz150" \
                       "\\brainscore_img_elec_time_70hz150\\"

    neural_responses = np.load(storage_location + 'neural_responses.npy')
    categories = np.load(storage_location + 'stimgroups.npy')
    category_mapping = {10: 'House', 20: 'Visage', 30: 'Animal', 40: 'Scene', 50: 'Tool', 90: 'Scramble'}

    # neural response is Images x Electrodes x Time: 300 x 11293 x 32
    # Categories is of size Images x 1 to represent the category of image
    # Categories can thus be thought of as the "stimuli"

    x_array = xr.DataArray(neural_responses, dims=('stimuli', 'electrodes', 'time_bin'),
                           coords={'stimuli': categories, 'electrodes': np.arange(11293),
                                   'time_bin': np.linspace(0, 1, num=32)})
