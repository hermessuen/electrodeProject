#!/bin/bash
export RESULTCACHING_HOME="$(pwd)/brain_score_results
export PYTHONPATH="$PYTHONPATH:$(pwd)/brain_score/brain-score
export PYTHONPATH="$PYTHONPATH:$(pwd)/brain_score/brainio_base
export PYTHONPATH="$PYTHONPATH:$(pwd)/brain_score/brainio_collection
export PYTHONPATH="$PYTHONPATH:$(pwd)/brain_score/brainio_contrib
export PYTHONPATH="$PYTHONPATH:$(pwd)/brain_score/candidate_modesl
export PYTHONPATH="$PYTHONPATH:$(pwd)/brain_score/model-tools
export PYTHONPATH="$PYTHONPATH:$(pwd)/tf-models/research/slim

