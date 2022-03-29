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
if [ -e SSAST-Base-Frame-400.pth ]
then
    echo "pretrained model already downloaded."
else
    wget https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1 -O SSAST-Base-Frame-400.pth
fi

pretrain_exp=unknown
pretrain_model=SSAST-Base-Frame-400

dataset=ciab
dataset_mean=-8.2096
dataset_std=6.1568
target_length=512
noise=True

bal=none
lr=1e-4
freqm=24
timem=96
mixup=0
epoch=10
batch_size=20
fshape=128
tshape=2
fstride=128
tstride=1

task=ft_avgtok
model_size=base
head_lr=1

pretrain_path=./${pretrain_model}.pth
base_exp_dir=./exp/test01-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}


exp_dir=${base_exp_dir}/all_train

train_data=./data/datafiles/ciab_train_data_${fold}.json
validation_data=./data/datafiles/ciab_validation_data_${fold}.json
standard_test_data=./data/datafiles/ciab_standard_test_data_${fold}.json
matched_test_data=./data/datafiles/ciab_matched_test_data_${fold}.json
long_test_data=./data/datafiles/ciab_long_test_data_${fold}.json

CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
--data-train ${train_data} --data-val ${validation_data} --data-standard-test ${standard_test_data} \
--data-matched-test ${matched_test_data} --data-long-test ${long_test_data} --exp-dir $exp_dir \
--label-csv ./data/ciab_class_labels_indices.csv --n_class 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrain False --pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc
done

python ./get_ciab_result.py --exp_path ${base_exp_dir}
