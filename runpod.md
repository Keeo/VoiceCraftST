# Installation commands for RUNPOD

- In order to run this on runpod make sure you are using `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`.
- Open port 5000 on Runpod interface or make yourself SSH tunel.

```bash
git clone https://github.com/Keeo/VoiceCraftST.git
cd VoiceCraftST

git submodule init
git submodule update

apt-get update && apt-get install --no-install-recommends -y  nano   python3-dev portaudio19-dev libportaudio2 libasound2-dev libportaudiocpp0 git python3 python3-pip make g++ ffmpeg wget espeak-ng &&     rm -rf /var/lib/apt/lists/*

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b && rm -f Miniconda3-latest-Linux-x86_64.sh

/root/miniconda3/bin/conda init && source ~/.bashrc

conda env create -f environment.yml
conda activate voicecraft

pip install -r ./api/requirements.txt
pip install -e "git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft"

mfa model download dictionary english_us_arpa && mfa model download acoustic english_us_arpa
mkdir /checkpoints /samples /users

python -m uvicorn api.server:app --reload --workers 1 --port 5000 --host 0.0.0.0 
```