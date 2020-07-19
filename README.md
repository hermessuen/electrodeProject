# Pre-requisites:

Prior to running any of the scripts in this repo, please make sure that brainio is installed according to this link:
https://github.com/brain-score/brain-score

# Format of the data

Brain score needs two types of data in order to run its tests: stimuli and assemblies. Stimuli are the images/audio/object that was presented during the experiment. Assemblies 
are the data response that you got, such as neuronal activations and their associated meta data. In order to be packaged into brainscore, The stimuli need to be be of class
"StimulusSet" which is a subset of the Pandas DataFrame class as defined here: https://github.com/brain-score/brainio_base/blob/master/brainio_base/stimuli.py#L4.
Assemblies need to be of type xarray: http://xarray.pydata.org/en/stable/

# Getting data into brainscore with these functions

"createAssembliesBrainScore" first calls two functions "load_responses" and "collect_stimuli" which take in as parameters the file path for the stimuli and the response files. It 
formats the Stimuli into the proper form to be cast into type "StimulusSet", and correctly labels all the dimensions for the xarray for the assemblies. It then calls functions that are defined 
in the "packaging_functions" folder to package the Stimuli and Assembly into a form that can be uploaded and integrated with brain score

In order to upload without errors, make sure that you have the proper permissions from the team that manages brainscore (AWS account that is authorized to dump data into this
particular bucket

# Data Analysis 
There are 4 data analysis functions:

1. calculate_mean_activations
2. category_impact
3. get_brodmann_areas
4. get_num_electrodes_in_area

The first one calculates the mean activations of the responses. You pass in a category, the Visual area of interest (V1-IT), and the time bin. Alternatively, you can pass in the parameter of "All" which means it will average across all electrodes. You can also pass in "All" for the time bin parameter which will return all of the time bin responses for a given electrode. 

"category_ impact" determines the impact of a category on a particular area of interest (brodmann area). You pass in the category and the area of interest. It then looks at all the electrodes in that region for that category and compares those activations to that of noise. It does this by running the first function on the category and then running the first function on noise. It then divides the two time bin vectors (1x32) to get a vector of factor increases for each time bin. It then returns this vector, as well as its average

"get_brodmann_areas" simply returns the number of unique brodmann areas in the data set

"get_num_electrodes_in_area" returns the number of electrodes contained in a given brodmann area. 


