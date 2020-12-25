cuda_device=$1


data_scale_list=(500 1000 1500 2000 3000 4000 5000 6000 7000 8000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 100000 100000 100000)

CUDA_VISIBLE_DEVICES=${cuda_device} nohup python run_disfluency_finetune_eval_each_electra_st.py --task_name disfluency --do_unlabel --do_lower_case --bert_model bert_model/electra_en_base/ --max_seq_length 128 --train_batch_size 64 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 30.0 --do_tagging --pretrain_model_dir self_supervised_model/ --pretrain_model_name pytorch_model.bin --model_name_or_path bert_model/electra_en_base/ --use_new_model --unlabel_size 500 --seed 1 --data_dir run_data/500/ --output_dir run_data/500/
wait

mv run_data/500/unlabel_results.txt run_data/500/pseudo.tsv

CUDA_VISIBLE_DEVICES=${cuda_device} nohup python run_disfluency_single_filter.py --task_name disfluency --do_eval --do_lower_case --data_dir run_data/500/ --bert_model bert_model/electra_en_base/ --max_seq_length 256 --train_batch_size 256 --learning_rate 5e-4 --num_train_epochs 600.0 --seed 1 --model_name_or_path bert_model/electra_en_base/ --use_new_model --pretrain_model_dir grammar_check_model/ --pretrain_model_name pytorch_model.bin --thre 0.5 --do_tagging
wait

for((i=0;i<29;i+=1)); 
do
data_scale=${data_scale_list[i]}
data_scale_next=${data_scale_list[((${i}+1))]}
CUDA_VISIBLE_DEVICES=${cuda_device} nohup python run_disfluency_finetune_eval_each_electra_st.py --task_name disfluency --do_train --do_eval --do_test --do_lower_case --bert_model bert_model/electra_en_base/ --max_seq_length 128 --train_batch_size 64 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 20.0 --do_tagging --pretrain_model_dir self_supervised_model/ --pretrain_model_name pytorch_model.bin --model_name_or_path bert_model/electra_en_base/ --use_new_model --unlabel_size 0 --seed ${data_scale} --data_dir run_data/${data_scale}/ --output_dir run_model/${data_scale}/
wait

CUDA_VISIBLE_DEVICES=${cuda_device} nohup python run_disfluency_finetune_eval_each_electra_st.py --task_name disfluency --do_unlabel --do_lower_case --bert_model bert_model/electra_en_base/ --max_seq_length 128 --train_batch_size 64 --gradient_accumulation_steps 1 --learning_rate 2e-5 --num_train_epochs 30.0 --do_tagging --pretrain_model_dir run_model/${data_scale} --pretrain_model_name pytorch_model.bin --model_name_or_path bert_model/electra_en_base/ --use_new_model --unlabel_size ${data_scale_next} --seed 1 --data_dir run_data/${data_scale}/ --output_dir run_data/${data_scale_next}/
wait

mv run_data/${data_scale_next}/unlabel_results.txt run_data/${data_scale_next}/pseudo.tsv
cp run_data/${data_scale}/dev.tsv run_data/${data_scale_next}/dev.tsv
cp run_data/${data_scale}/test.tsv run_data/${data_scale_next}/test.tsv
cp run_data/${data_scale}/unlabel.tsv run_data/${data_scale_next}/unlabel.tsv


CUDA_VISIBLE_DEVICES=${cuda_device} nohup python run_disfluency_single_filter.py --task_name disfluency --do_eval --do_lower_case --data_dir run_data/${data_scale_next}/ --bert_model bert_model/electra_en_base/ --max_seq_length 256 --train_batch_size 256 --learning_rate 5e-4 --num_train_epochs 600.0 --seed 1 --model_name_or_path bert_model/electra_en_base/ --use_new_model --pretrain_model_dir grammar_check_model/ --pretrain_model_name pytorch_model.bin --thre 0.5 --do_tagging

done



