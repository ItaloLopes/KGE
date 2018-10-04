#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load gcc
module load python
module load cuda-toolkit/8.0.44
module load cudnn/5.1

export THEANO_FLAGS='device=cuda0'

cd /mnt/nfs/home/ItaloLO/KGE/
python3 run.py evaluation --model Complex --data datasets/wn18.txt --k 200 --epoch 1000 --folds 5

