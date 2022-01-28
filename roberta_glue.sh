export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for dense_prune_method in topK
do
for mask_lr in 0.1
do
python glue_pruning.py \
--model_name_or_path roberta-base \
--task_name mnli \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 128 \
--learning_rate 3e-4 \
--warmup_ratio 0.06 \
--num_train_epochs 10 \ 
--logging_steps 100 \
--warmup_steps 5000 \
--seed 0 \
--weight_decay 0.0 \
--mask_lr $mask_lr \
--report_to wandb \
--dense_pruning_method $dense_prune_method \
--dense_pruning_submethod $dense_pruning_submethod \
--attention_pruning_method disabled \
--regularization disabled \
--prune_leftover 0.1 \
--apply_parallel_adapter \
--parallel_adapter_size 16 
done
done
#--output_dir $dir/$model_name_or_path \
#--apply_lora \
#--lora_r 8 \
#--lora_alpha 16 \
#--apply_adapter \
#--adapter_type houlsby \
#--adapter_size 16