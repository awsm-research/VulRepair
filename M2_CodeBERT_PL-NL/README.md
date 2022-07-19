## Replicate M2 Model

Please first download the model "model.bin" through the following commands:
```
cd M2_CodeBERT_PL-NL
cd saved_models
cd checkpoint-best-loss
gdown https://drive.google.com/uc?id=1iDmeZgIZgI4i3-IQzXVwsBLlMuF41YFc
cd ../../..
```

To reproduce the results of M2 model, run the following commands **(Inference only)**:
```
cd M2_CodeBERT_PL-NL
python codebert_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 
```

To retrain the M2 model from scratch, run the following commands **(Training only)**:
```
cd M2_CodeBERT_PL-NL
python codebert_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
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
