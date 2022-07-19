## Replicate VulRepair (M1 Model)

To reproduce the results of our VulRepair (M1 model), run the following commands **(Inference only)**:
```
cd M1_VulRepair_PL-NL
python vulrepair_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=MickyMike/VulRepair \
    --model_name_or_path=MickyMike/VulRepair \
    --do_test \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --num_beams=50 \
    --eval_batch_size 1
```
Note. please adjust the "num_beams" parameters accordingly to obtain the results we present in the discussion section. (i.e., num_beams= 1, 2, 3, 4, 5, 10)

To retrain the M1 model from scratch, run the following commands **(Training only)**:
```
cd M1_VulRepair_PL-NL
python vulrepair_main.py \
    --model_name=model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
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
