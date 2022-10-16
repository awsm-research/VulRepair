## Replicate M3 Model

Please first download the model "model.bin" through the following commands:
```
cd M3_T5_no_pretrain_subword
cd saved_models
cd checkpoint-best-loss
gdown https://drive.google.com/uc?id=17Au8hroq5zHjMSEGuXBLUnmA5D-7sJda
cd ../../..
```

To reproduce the results of M3 model, run the following commands **(Inference only)**:
```
cd M3_T5_no_pretrain_subword
python t5_no_pretraining_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=./tokenizer/ \
    --config_name=t5-base \
    --do_test \
    --test_data_file=../data/fine_tune_data/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1
```

To retrain the M3 model from scratch, run the following commands **(Training only)**:
```
cd M3_T5_no_pretrain_subword
python t5_no_pretraining_main.py \
    --model_name=model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=./tokenizer/ \
    --config_name=t5-base \
    --do_train \
    --train_data_file=../data/fine_tune_data/train.csv \
    --eval_data_file=../data/fine_tune_data/val.csv \
    --test_data_file=../data/fine_tune_data/test.csv \
    --epochs 75 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```
