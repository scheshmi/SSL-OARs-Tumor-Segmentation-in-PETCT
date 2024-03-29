# SSL-OARs-Tumor-Segmentation-in-PETCT
Self-Supervised Learning with Swin UNETR for Segmentation of Organs at Risk and Tumor in PET/CT Images

# Install Dependencies
Install dependencies using
```python 
pip install -r requirements.txt
```
# Preprocessing
Before pretraining and fine-tuning, data (PET and CT) should be preprocessed:
```python
python preprocess.py --in_dir=<Input-directory(PET and CT)> --out_dir=<Output-directory>
```
# Pre-Training 
Pre-Train Swin UNETR encoder on unlabeled data
```python 
python main.py --exp=<Experiment Name> --in_channels=2 --data_dir=<Data-Path> --json_list=<Json List Path> \
--lr=6e-6 --lrdecay --batch_size=<Batch Size> --num_steps=<Number of Steps>
```
# Fine-Tuning 
Fine-Tuning Swin UNETR on labeled data
```python 
python main.py --exp=<Experiment Name> --data_dir=<Data-Path> --json_list=<Json List Path> --in_channels=2 --out_channels=12 \
--pretrained_model_name=<Pretrained Encoder Name> --batch_size=<Batch Size> --max_epochs=<Epochs> --use_ssl_pretrained \
--ssl_pretrained_path=<Pretrained Model Path> --use_checkpoint
```
# Evaluation
Evaluating Swin UNETR 
```python 
python test.py --pretrained_dir=<Pretrained Model Path> --data_dir=<Data-Path> --exp_name=<Experiment Name> \
--json_list=<Json List Path> --pretrained_model_name=<Pretrained Model Name> --save 
```
# Acknowledgement
Models implementation and SSL pipeline are based on [MONAI](https://github.com/Project-MONAI/MONAI) and [This](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) repositories. 
