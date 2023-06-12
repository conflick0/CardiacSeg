sudo apt update && \
sudo apt install ffmpeg libsm6 libxext6  -y && \
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && \
pip install "monai[nibabel, tqdm, einops]" && \
pip install monailabel && \
pip install -U openmim && \
mim install -U mmcv-full && \
pip install timm==0.6.13 && \
pip install tensorboard && \
pip install ml-collections && \
pip install scikit-learn && \
pip install pandas==1.5.3 && \
pip install matplotlib && \
pip install ray==2.4.0 && \
pip install lion-pytorch && \
pip install torchsummaryX && \
pip install --user ipykernel && \
python -m ipykernel install --user --name base