## Replicate M6 Model

Please first download the model "model.bin" through the following commands:
```
cd M6_BERT_base_NL
cd saved_models
cd checkpoint-best-loss
gdown https://drive.google.com/uc?id=1y_EXyNvkQCFs-Zr2cULZRH-hpg7fpjru
cd ../../..
```

To reproduce the results of M6 model, run the following commands **(Inference only)**:
```
cd M6_BERT_base_NL
python roberta_base_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=roberta-base \
    --model_name_or_path=roberta-base \
    --do_test \
    --test_data_file=../data/fine_tune_data/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 
```

To retrain the M6 model from scratch, run the following commands **(Training only)**:
```
cd M6_BERT_base_NL
python roberta_base_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=roberta-base \
    --model_name_or_path=roberta-base \
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
