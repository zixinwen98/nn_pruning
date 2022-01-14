export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
output_dir="./pruning_roberta_base_mnli"
for dense_prune_method in topK sigmoied_threshold uniqueness
do
for dense_pruning_submethod in 1d_alt 1d_alt
do
python glue_pruning.py \
--model_name_or_path gpt2 \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 128 \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 10 \
--output_dir $output_dir/$model_name_or_path \
--logging_steps 100 \
--warmup_steps 5000 \
--seed 0 \
--weight_decay 0.0 \
--report_to wandb \
--dense_pruning_method $dense_prune_method \
--dense_pruning_submethod $dense_pruning_submethod \
--attention_pruning_method disabled \
--regularization disabled \
--prune_leftover 0.05
done
done