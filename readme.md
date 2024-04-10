# Improving the classification of multiple sclerosis and cerebral small vessel disease with interpretable transfer attention neural network

## Abstract

**Background**: As an autoimmune-mediated inflammatory demyelinating disease of the central nervous system, multiple sclerosis (MS) is often confused with cerebral small vessel disease (cSVD), which is a regional pathological change in brain tissue with unknown pathogenesis. This is due to their similar clinical presentations and imaging manifestations. That misdiagnosis can significantly increase the occurrence of adverse events. Delayed or incorrect treatment is one of the most important causes of MS progression. Therefore, the development of a practical diagnostic imaging aid could significantly reduce the risk of misdiagnosis and improve patient prognosis.
**Method**: We propose an interpretable deep learning model that differentiates MS and cSVD using T2-weighted fluid-attenuated inversion recovery (FLAIR) images. Our model uses a pre-trained convolutional neural network to extract the texture features of the images and achieves more robust feature learning through two attention modules. The attention maps provided by the attention modules provide model interpretation to validate model learning and reveal more information to physicians. Finally, the proposed model is trained end-to-end using focal loss to reduce the influence of class imbalance.
**Results**: The model was validated using clinically diagnosed MS (n=112) and cSVD (n=321) patients from the Beijing Tiantan Hospital database. The performance of the proposed model was better than that of two commonly used deep learning approaches, with a mean balanced accuracy of 86.06% and a mean area under the receiver operating characteristic curve of 98.78%. Moreover, the generated attention heat maps showed that the proposed model could focus on the lesion signatures in the image.
**Conclusions**: The proposed model provides a practical diagnostic imaging aid for the use of routinely available imaging techniques such as magnetic resonance imaging to classify MS and cSVD, as well as a generalizable approach for linking deep learning to human brain disease.

## Usage

1. Prepare the enviroment. We recommend using the `mamba` or `conda`.

    ```bash
    mamba env create -f ./env_hist.yaml
    ```

2. Constract a index file, which is a csv file and must include absolute image filename, label.
3. Run the script to training model.

    ```bash
    python ./scripts/train.py --index_file path/to/index/file --save_root path/to/save/root --model cnn2datt ---device cuda:0
    ```
    Details of the `train.py` script usage can be found using `python ./scripts/train.py --help`.

```
usage: train.py [-h] [--index_file INDEX_FILE] [--save_root SAVE_ROOT]
                [--cv CV] [--valid_size VALID_SIZE] [--test_size TEST_SIZE]
                [--seed SEED] [--slice_index SLICE_INDEX SLICE_INDEX]
                [--target_shape TARGET_SHAPE TARGET_SHAPE TARGET_SHAPE]
                [--model {cnn2datt,cnn3d,sklearn_svc,sklearn_rf}]
                [--backbone BACKBONE] [--no_backbone_pretrained]
                [--backbone_feature_index BACKBONE_FEATURE_INDEX]
                [--backbone_freeze] [--no_satt]
                [--satt_hiddens [SATT_HIDDENS [SATT_HIDDENS ...]]]
                [--satt_acts [SATT_ACTS [SATT_ACTS ...]]] [--no_satt_bn]
                [--satt_dp SATT_DP] [--no_iatt] [--iatt_hidden IATT_HIDDEN]
                [--iatt_bias] [--iatt_temperature IATT_TEMPERATURE]
                [--w_kl_satt W_KL_SATT]
                [--mlp_hiddens [MLP_HIDDENS [MLP_HIDDENS ...]]]
                [--mlp_act MLP_ACT] [--no_mlp_bn] [--mlp_dp MLP_DP]
                [--loss_func {ce,focal}] [--focal_alpha FOCAL_ALPHA]
                [--focal_gamma FOCAL_GAMMA] [--device DEVICE]
                [--nepoches NEPOCHES] [--learning_rate LEARNING_RATE]
                [--no_modelcheckpoint] [--no_early_stop]
                [--early_stop_patience EARLY_STOP_PATIENCE] [--lr_schedual]
                [--lr_sch_factor LR_SCH_FACTOR]
                [--lr_sch_patience LR_SCH_PATIENCE]
                [--monitor_metric {bacc,acc,auc}]
                [--message_level MESSAGE_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --index_file INDEX_FILE
  --save_root SAVE_ROOT
  --cv CV
  --valid_size VALID_SIZE
  --test_size TEST_SIZE
  --seed SEED
  --slice_index SLICE_INDEX SLICE_INDEX
  --target_shape TARGET_SHAPE TARGET_SHAPE TARGET_SHAPE
                        necessary for nilearn
  --model {cnn2datt,cnn3d,sklearn_svc,sklearn_rf}
  --backbone BACKBONE
  --no_backbone_pretrained
  --backbone_feature_index BACKBONE_FEATURE_INDEX
  --backbone_freeze
  --no_satt
  --satt_hiddens [SATT_HIDDENS [SATT_HIDDENS ...]]
  --satt_acts [SATT_ACTS [SATT_ACTS ...]]
  --no_satt_bn
  --satt_dp SATT_DP
  --no_iatt
  --iatt_hidden IATT_HIDDEN
  --iatt_bias
  --iatt_temperature IATT_TEMPERATURE
  --w_kl_satt W_KL_SATT
  --mlp_hiddens [MLP_HIDDENS [MLP_HIDDENS ...]]
  --mlp_act MLP_ACT
  --no_mlp_bn
  --mlp_dp MLP_DP
  --loss_func {ce,focal}
  --focal_alpha FOCAL_ALPHA
  --focal_gamma FOCAL_GAMMA
  --device DEVICE
  --nepoches NEPOCHES
  --learning_rate LEARNING_RATE
  --no_modelcheckpoint
  --no_early_stop
  --early_stop_patience EARLY_STOP_PATIENCE
  --lr_schedual
  --lr_sch_factor LR_SCH_FACTOR
  --lr_sch_patience LR_SCH_PATIENCE
  --monitor_metric {bacc,acc,auc}
  --message_level MESSAGE_LEVEL
                        2 means all messages, 1 means all messages but epoch
                        print, 0 means no message.
```