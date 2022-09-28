'''
File to train ssast on symptoms classification on just covid negative individuals
then evaluate on covid detection
'''
set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source ../../../venvssast/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir exp

if [ -e SSAST-Base-Frame-400.pth ]
then
    echo "pretrained model already downloaded."
else
    wget https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1 -O SSAST-Base-Frame-400.pth
fi

pretrain_exp=unknown
pretrain_model=SSAST-Base-Frame-400

for dataset in "ciab_sentence" "ciab_three_cough" "ciab_cough" "ciab_ha_sound"; do
	echo $dataset
	if [ "$dataset" = "ciab_three_cough" ]
	then
		dataset_mean=-8.2274
		dataset_std=6.1628
		dir_name=audio_three_cough_url
		symp_dir_name=three_cough_url
	elif [ "$dataset" = "ciab_sentence" ]
	then	
		dataset_mean=-7.8505
		dataset_std=5.4594
		dir_name=audio_sentence_url
		symp_dir_name=sentence_url
	elif [ "$dataset" = "ciab_ha_sound" ]
	then
		dataset_mean=-8.8451
		dataset_std=5.9847
		dir_name=audio_ha_sound_url
		symp_dir_name=exhalation_url
	elif [ "$dataset" = "ciab_cough" ]
	then
		dataset_mean=-7.7998
		dataset_std=6.3570
		dir_name=audio_cough_url
		symp_dir_name=cough_url
	else
		echo "not working"
	fi
	target_length=512
	noise=True

	bal=none
	lr=1e-4
	freqm=24
	timem=96
	mixup=0
	epoch=20
	batch_size=20
	fshape=128
	tshape=2
	fstride=128
	tstride=1

	task=ft_avgtok
	model_size=base
	head_lr=1
	fold=1
	pretrain_path=./${pretrain_model}.pth
	base_exp_dir=./exp/symptoms/${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}-standard-train

	exp_dir=${base_exp_dir}/fold${fold}
	train_data=./data/datafiles/${symp_dir_name}/ciab_train_symp_data_${fold}.json
	validation_data=./data/datafiles/${symp_dir_name}/ciab_val_symp_data_${fold}.json
	test_data=./data/datafiles/${dir_name}/ciab_test_data_${fold}.json
	matched_test_data=./data/datafiles/${dir_name}/ciab_matched_test_data_${fold}.json
	long_test_data=./data/datafiles/${dir_name}/ciab_long_test_data_${fold}.json
	matched_long_test_data=./data/datafiles/${dir_name}/ciab_long_matched_data_${fold}.json
	naive_train_data=./data/datafiles/${symp_dir_name}/ciab_naive_train_symp_data_${fold}.json
	naive_validation_data=./data/datafiles/${symp_dir_name}/ciab_naive_val_symp_data_${fold}.json
	naive_test_data=./data/datafiles/${dir_name}/naive_test_${fold}.json

	# standard train
	CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
	--data-train ${train_data} --data-val ${validation_data} --data-test ${test_data} \
	--data-matched-test ${matched_test_data} --data-long-test ${long_test_data} \
	--data-long-matched ${matched_long_test_data} \
	--label-csv ./data/ciab_class_labels_indices.csv --n_class 2 --exp-dir $exp_dir \
	--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
	--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
	--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
	--model_size ${model_size} --adaptschedule False \
	--pretrain False --pretrained_mdl_path ${pretrain_path} \
	--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
	--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
	--lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics mAP


	# naive method
	base_exp_dir=./exp/symptoms/${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}-naive
	exp_dir=${base_exp_dir}/fold${fold}
	CUDA_CACHE_DISABLE=1 python -W ignore ../../run.py --dataset ${dataset} \
	--data-train ${naive_train_data} --data-val ${naive_validation_data} \
	--data-test ${naive_test_data} \
	--exp-dir $exp_dir \
	--label-csv ./data/ciab_class_labels_indices.csv --n_class 2 \
	--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
	--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
	--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
	--model_size ${model_size} --adaptschedule False \
	--pretrain False --pretrained_mdl_path ${pretrain_path} \
	--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
	--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
	--lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics mAP
done

