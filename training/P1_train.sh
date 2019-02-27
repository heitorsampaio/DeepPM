#!/bin/bash -l
#SBATCH -J  train
#SBATCH -o train-%j.out
#SBATCH -p gpu3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 20G
#SBATCH --gres gpu:1
#SBATCH -t 2-00:00:00

## Load Needed Modules
##module load cuda/cuda-8.0

GLOBAL_PATH='/Users/heitorsampaio/Google_Drive/Projetos/Protein_DeepLearning/DeepPM';

datadir=$GLOBAL_PATH/datasets/D1_SimilarityReduction_dataset
outputdir=$GLOBAL_PATH/test/output_nb_layers_5_fc_hidden_1000
echo "#################  Training on inter 15"

nb_filters=10
nb_layers=5
fc_hidden=1000

## Test Theano
THEANO_FLAGS=floatX=float32,device=cpu python3 $GLOBAL_PATH/training/training_main.py 15 $nb_filters $nb_layers nadam '6_10' $fc_hidden 30 50 3 $datadir $outputdir
## Test Theano
THEANO_FLAGS=floatX=float32,device=cpu python3 $GLOBAL_PATH/training/predict_main.py 15 $nb_filters $nb_layers nadam '6_10' $fc_hidden 30 50 3 $datadir $outputdir
