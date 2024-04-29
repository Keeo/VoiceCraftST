FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Set label for the docker image description
LABEL description="Docker image VoiceCraft"

# Install required packages (avoid cache to reduce image size)
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3-dev portaudio19-dev libportaudio2 libasound2-dev libportaudiocpp0 \
    git python3 python3-pip make g++ ffmpeg wget espeak-ng && \
    rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version

COPY VoiceCraft/environment.yml environment.yml
COPY api/requirements.txt requirements.txt

RUN conda env create -f environment.yml
RUN conda run --no-capture-output -n voicecraft pip install -r requirements.txt
RUN conda run --no-capture-output -n voicecraft pip install -e "git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft"

RUN conda run --no-capture-output -n voicecraft mfa model download dictionary english_us_arpa
RUN conda run --no-capture-output -n voicecraft mfa model download acoustic english_us_arpa

COPY . .

# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "voicecraft", "python", "-m", "uvicorn", "api/server:app", "--reload"]
