set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
source ../../../venvssast/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir exp

# prep ciab dataset and download the pretrained model
if [ -e data/datafiles ]
then
    echo "ciab already downloaded and processed."
else
    echo "preparing the ciab dataset"
    python prep_ciab.py
fi
if [ -e SSAST-Base-Patch-400.pth ]
then
    echo "pretrained model already downloaded."
else
wget https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1 -O SSAST-Base-Patch-400.pth
fi

pretrain_exp=unknown
pretrain_model=SSAST-Base-Patch-400

dataset=ciab_three_cough
if [ "$dataset" == ciab_three_cough ]
then
	dataset_mean=-8.2096
	dataset_std=6.1568
elif [ "$dataset" == ciab_sentence ]
then	
	dataset_mean=-7.8382
	dataset_std=5.4489
elif [ "$dataset" == ciab_ha_sound ]
then
	dataset_mean=-8.8335
	dataset_std=5.9809
else
	dataset_mean=-7.7893
	dataset_std=6.3424
fi
target_length=512
noise=True

bal=none
lr=1e-4
freqm=24
timem=96
mixup=0
epoch=20
batch_size=18
fshape=16
tshape=16
fstride=10
tstride=10
logger=wandb

task=ft_cls
model_size=base
head_lr=1
fold=1
pretrain_path=./${pretrain_model}.pth
base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}-standard-train-2

echo 'now process fold'${fold}

exp_dir=${base_exp_dir}/fold${fold}

train_data=./data/datafiles/audio_three_cough_url/ciab_train_data_${fold}.json
validation_data=./data/datafiles/audio_three_cough_url/ciab_validation_data_${fold}.json
standard_test_data=./data/datafiles/audio_three_cough_url/ciab_standard_test_data_${fold}.json
matched_test_data=./data/datafiles/audio_three_cough_url/ciab_matched_test_data_${fold}.json
matched_train_data=./data/datafiles/audio_three_cough_url/ciab_matched_train_data_${fold}.json
matched_validation_data=./data/datafiles/audio_three_cough_url/ciab_matched_validation_data_${fold}.json
long_test_data=./data/datafiles/audio_three_cough_url/ciab_long_test_data_${fold}.json
naive_train_data=./data/datafiles/audio_three_cough_url/naive_train_${fold}.json
naive_validation_data=./data/datafiles/audio_three_cough_url/naive_validation_${fold}.json
naive_test_data=./data/datafiles/audio_three_cough_url/naive_test_${fold}.json
big_train_data=./data/datafiles/audio_three_cough_url/big_train_${fold}.json
big_validation_data=./data/datafiles/audio_three_cough_url/big_validation_${fold}.json

# standard train
CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${train_data} --data-val ${validation_data} --data-standard-test ${standard_test_data} \
--data-matched-test ${matched_test_data} --exp-dir $exp_dir --data-long-test ${long_test_data} \
--label-csv ./data/ciab_class_labels_indices.csv --n_class 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrain False --pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics mAP \
--wandb ${logger}

#  matched train
base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}-matched-train-2
exp_dir=${base_exp_dir}/fold${fold}
CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${matched_train_data} --data-val ${matched_validation_data} \
--data-standard-test ${standard_test_data} --data-matched-test ${matched_test_data} --exp-dir $exp_dir \
--label-csv ./data/ciab_class_labels_indices.csv --n_class 2 --data-long-test ${long_test_data} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrain False --pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics mAP \
--wandb ${logger}

# naive method
base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}-naive-final-2
exp_dir=${base_exp_dir}/fold${fold}
CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${naive_train_data} --data-val ${naive_validation_data} \
--data-standard-test ${naive_test_data} \
--exp-dir $exp_dir \
--label-csv ./data/ciab_class_labels_indices.csv --n_class 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrain False --pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics mAP \
--wandb ${logger}

## train = train + long  method
base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}-big-final-2
exp_dir=${base_exp_dir}/fold${fold}
CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${big_train_data} --data-val ${big_validation_data} \
--data-standard-test ${standard_test_data} --data-matched-test ${matched_test_data} \
--exp-dir $exp_dir \
--label-csv ./data/ciab_class_labels_indices.csv --n_class 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrain False --pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics mAP \
--wandb ${logger}
