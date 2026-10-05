# MHAFR-DDI
MHAFR-DDI: A multimodal hierarchical attention fusion and relation-aware architecture for drug–drug interaction event prediction
## Pretrained model
You can find our pretrained MHAFR-DDI in `/save`.


## Data
The first dataset originates from the study by Deng et al. [1], containing 572 drugs, 65 DDI types, and a total of 37,264 DDI instances. The second dataset, from Ryu et al. [2], includes 1,700 drugs, 86 DDI types, and 191,570 DDI records in total.

[1] Y. Deng, X. Xu, Y. Qiu, J. Xia, W. Zhang, S. Liu, A multimodal deep learning framework for predicting drug–drug interaction events, Bioinformatics, 36 (2020) 4316-4322 

[2] J.Y. Ryu, H.U. Kim, S.Y. Lee, Deep learning improves prediction of drug–drug and drug–food interactions, Proceedings of the national academy of sciences, 115 (2018) E4304-E4311


## Environment
You can create a conda environment for MHAFR-DDI by `conda env create -f environment.yml`.


## Pretrain and Finetune

You can pretrain MHAFR-DDI by `python pretrain_MHAFR-DDI.py`. You can finetune the pretrained MHAFR-DDI for drug interaction prediction tasks by `python ddi_dnn-z.py`.


