## Replicate M9 Model

Please first download the model "model.bin" through the following commands:
```
cd M9_CodeBERT_word_level
cd saved_models
cd checkpoint-best-loss
gdown https://drive.google.com/uc?id=10hLd4kkOffstveBqzxkzPP4_zv-5E_gy
cd ../../..
```

To reproduce the results of M9 model, run the following commands **(Inference only)**:
```
cd M9_CodeBERT_word_level
python codebert_wordlevel_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --config_name=roberta-base \
    --do_test \
    --test_data_file=../data/fine_tune_data/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 
```

To retrain the M9 model from scratch, run the following commands **(Training only)**:
```
cd M9_CodeBERT_word_level
python codebert_wordlevel_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --config_name=roberta-base \
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
