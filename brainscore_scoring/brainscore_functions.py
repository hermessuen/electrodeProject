import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.utils import LazyLoad
import numpy as np
from brainio_base.assemblies import walk_coords
from brainscore.metrics import Score


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 50



def AruKuzovkin2018PLS():
    assembly_repetition = LazyLoad(lambda: load_assembly(average_repetitions=False))
    assembly = LazyLoad(lambda: load_assembly(average_repetitions=True))
    similarity_metric = CrossRegressedCorrelation(
        regression=pls_regression, correlation=pearsonr_correlation
    )

    ceiler = InternalConsistency()

    x = assembly['region']
    idx = x.data == 20
    new_assembly = assembly[:, idx, :]

    new_assembly = type(new_assembly)(new_assembly.values, coords={coord: (dims, values if coord != 'region' else ['IT'] * len(new_assembly['region'])) for
                                   coord, dims, values in walk_coords(new_assembly)}, dims=new_assembly.dims)

    ceiling = Score([1, np.nan])
    return NeuralBenchmark(identifier=f'aru.Kuzovkin2018-pls', version=1, assembly=new_assembly,
                           similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, ceiling_func=lambda: ceiler,
                           parent='Kamila', number_of_trials=1)

    #return NeuralBenchmark(identifier=f'aru.Kuzovkin2018-pls', version=1, assembly=new_assembly, similarity_metric=similarity_metric,
                           #visual_degrees=VISUAL_DEGREES, ceiling_func=lambda: ceiler(assembly_repetition),
                           #parent='Kamila', number_of_trials=1)



def load_assembly(average_repetitions):
    assembly = brainscore.get_assembly(name='aru.Kuzovkin2018')
    if average_repetitions:
        assembly = average_repetition(assembly)

    return assembly

