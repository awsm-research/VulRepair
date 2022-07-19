## Replicate M5 Model

Please first download the model "model.bin" through the following commands:
```
cd M5_BERT_no_pretrain_subword
cd saved_models
cd checkpoint-best-loss
gdown https://drive.google.com/uc?id=1RuAppue9ff7xHiOyI7tPQEvoOTnRvh11
cd ../../..
```

To reproduce the results of M5 model, run the following commands **(Inference only)**:
```
cd M5_BERT_no_pretrain_subword
python roberta_no_pretraining_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=./tokenizer/ \
    --config_name=roberta-base \
    --do_test \
    --test_data_file=../data/fine_tune_data/cve_fixes_test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1
```

To retrain the M5 model from scratch, run the following commands **(Training only)**:
```
cd M5_BERT_no_pretrain_subword
python roberta_no_pretraining_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=./tokenizer/ \
    --config_name=roberta-base \
    --do_train \
    --train_data_file=../data/fine_tune_data/cve_fixes_train.csv \
    --eval_data_file=../data/fine_tune_data/cve_fixes_val.csv \
    --test_data_file=../data/fine_tune_data/cve_fixes_test.csv \
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
