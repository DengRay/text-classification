

export GLUE_DIR=C:\\Users\\pc\\Desktop\\bert-Chinese-classification-task-master
export BERT_BASE_DIR=C:\\Users\\pc\\Desktop\\bert-Chinese-classification-task-master\\chinese_L-12_H-768_A-12
export BERT_PYTORCH_DIR=C:\\Users\\pc\\Desktop\\bert-Chinese-classification-task-master\\chinese_L-12_H-768_A-12

python run_classifier_word.py \
  --task_name NEWS \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/data/ \
  --vocab_file $BERT_BASE_DIR\\vocab.txt \
  --bert_config_file $BERT_BASE_DIR\\bert_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR\\pytorch_model.bin \
  --max_seq_length 256 \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./newsAll_output/ \
  --local_rank -1
