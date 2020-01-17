CUDA_VISIBLE_DEVICES=${1} \
python main_stream.py \
--stage='train_test_fuse' \
--pretrain_dataset='UCF101' \
--mode='temporal' \
--method='main_stream' \
--method_temporal='main_stream' \
--log_dir='/media/yang/E/ssad_experiments/64s/logs/' \
--model_dir='/media/yang/E/ssad_experiments/64s/models/' \
--results_dir='/media/yang/E/ssad_experiments/64s/results/'
