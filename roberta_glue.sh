export CUDA_VISIBLE_DEVICES=0,1,2,3
output_dir="./pruning_roberta_base_mnli"
for dense_prune_method in topK magnitude threshold sigmoied_threshold
do
python glue_pruning.py \
--model_name_or_path roberta-base \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 128 \
--learning_rate 3e-5 \
--num_train_epochs 10 \
--output_dir $output_dir/$model_name_or_path \
--logging_steps 100 \
--warmup_steps 5000 \
--seed 0 \
--weight_decay 0.0 \
--report_to wandb \
--dense_pruning_method $dense_prune_method \
--dense_pruning_submethod 1d_alt \
--attention_pruning_method disabled \
--regularization l1 \
--prune_leftover 0.15
done