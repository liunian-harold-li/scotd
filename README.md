# SCOTD

Repo for the paper [Symbolic Chain of Thought Distillation](https://arxiv.org/pdf/2306.14050.pdf).

## Environment

<!-- docker run --name harold_test  -it --runtime=nvidia --ipc=host -v /local2/harold/:/local2/harold pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel -->

```
pip install transformers==4.21.1 tensorboard yacs tensorboardX accelerate==0.14.0 pandas wandb openai sentencepiece datasets==1.18.3 torchtext deepspeed matplotlib seaborn markupsafe==2.0.1 sentence_transformers fastcluster
```
Please run ```accelerate config``` to set up the accelerate environment. We used DeepSpeed (default config and ZeRO optimization level 2) and fp16 to train the models on a A6000.

## Data
Please download the data and put them under "DATA" folder. Data is available at [here](https://drive.google.com/drive/folders/1u8ecmIzH88V_z4mzs8mSNk5T3NAnoe19?usp=share_link).
```
DATA
  csqa_30x.json
  quarel_30x.json
  openbook_30x.json
```

## Running the Code

### Training

#### CommonsenseQA
```
export GPU_ID=0
bash _scripts/train_single_task.bash \
  $GPU_ID 1720 \
  configs/train.yaml \
  CSQA_30x configs_data/complete/commonsenseqa.py DATA/csqa_30x.json \
  RUN_UID csqa
```

#### QuaRel
```
export GPU_ID=0
bash _scripts/train_single_task.bash \
  $GPU_ID 1721 \
  configs/train.yaml \
  QuaRel_30x configs_data/complete/quarel.py DATA/quarel_30x.json \
  RUN_UID quarel
```
#### OpenBookQA
```
export GPU_ID=0
bash _scripts/train_single_task.bash \
  $GPU_ID 1722 \
  configs/train.yaml \
  OB_30x configs_data/complete/openbook.py DATA/openbook_30x.json \
  RUN_UID openbook
```

### Evaluation
```
export DIR_EVAL=OUTPUTS/{RUN_UID}
export CONFIG=configs_data/complete/{task}.py
export MODEL_EPOCH=best
export GPU_ID=0
bash _scripts/eval_single_task.bash $GPU_ID \
    DATA.CONFIG $CONFIG \
    TRAIN.SAMPLE_ROUNDS 1 \
    POLICY_MODEL.TEMPERATURE 0.0 \
    POLICY_MODEL.DO_SAMPLE False
```

Some pre-trained checkpoints are available at [here](https://drive.google.com/drive/folders/1u8ecmIzH88V_z4mzs8mSNk5T3NAnoe19?usp=share_link).