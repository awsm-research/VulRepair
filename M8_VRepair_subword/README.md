## Replicate M8 Model

**Step 1: PLEASE FIRST INSTALL THE CUSTOM Transformers Library USING THE FOLLOWING COMMANDS:**
```
cd M8_VRepair_subword
cd transformers
pip install .
cd ../..
```

**ATTENTION! THIS CUSTOM LIBRARY ONLY WORKS FOR THIS MODEL (i.e., M8_VRepair_subword) AND DOES NOT WORK FOR OTHER MODELS**

After replicating this model, **you should uninstall the custom Transformers Library via "pip uninstall transformers"**

Then, **reinstall original Transformers Library via "pip install transformers" which works for all other models**

**Misuse the library may lead to inaccurate replication, PLEASE BE CAREFUL.**

**Step 2:** Then, download the model "model.bin" through the following commands:
```
cd M8_VRepair_subword
cd saved_models
cd checkpoint-best-loss
gdown https://drive.google.com/uc?id=1uJDqx8mqIA4R5S1F1OkxSk_pWw_Cl74V
cd ../../..
```

**Step 3:** To reproduce the results of M8 model, run the following commands **(Inference only)**:
```
cd M8_VRepair_subword
python vrepair_subword_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=Salesforce/codet5-base \
    --config_name=t5-base \
    --do_test \
    --test_data_file=../data/fine_tune_data/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 
```

**Step 3:** To retrain the M8 model from scratch, run the following commands **(Training only)**:
```
cd M8_VRepair_subword
python vrepair_subword_main.py \
    --model_name=model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=Salesforce/codet5-base \
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
