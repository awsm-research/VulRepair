## Replicate M4 Model

Please first download the model "model.bin" through the following commands:
```
cd M4_T5_base_NL
cd saved_models
cd checkpoint-best-loss
gdown https://drive.google.com/uc?id=169YMd3y7J9YlCg8_qIAUtpuIQSlfo_fW
cd ../../..
```

To reproduce the results of M4 model, run the following commands **(Inference only)**:
```
cd M4_T5_base_NL
python t5_base_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=t5-base \
    --model_name_or_path=t5-base \
    --do_test \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 
```

To retrain the M4 model from scratch, run the following commands **(Training only)**:
```
cd M4_T5_base_NL
python t5_base_main.py \
    --model_name=model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=t5-base \
    --model_name_or_path=t5-base \
    --do_train \
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
