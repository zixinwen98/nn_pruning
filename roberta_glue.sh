output_dir="./pruning_roberta_base_mnli"
for dense_prune_method in topK magnitude threshold sigmoied_threshold
do
CUDA_VISIBLE_DEVICES=0 python glue_pruning.py \
--model_name_or_path roberta-base \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 3e-4 \
--num_train_epochs 5 \
--output_dir $output_dir/$model_name_or_path \
--logging_steps 30 \
--seed 0 \
--weight_decay 0.1 \
--report_to wandb \
--dense_pruning_method $dense_prune_method \
--attention_pruning_method disabled \
--regularization disabled 
done