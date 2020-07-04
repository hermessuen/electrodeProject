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

