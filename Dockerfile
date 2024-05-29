FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,utility,video

# install utilities
RUN apt update && \
    apt install git gcc vim net-tools ffmpeg portaudio19-dev firmware-sof-signed\
    alsa-base alsa-utils sox    \
    -y

WORKDIR /workspace
COPY . /workspace
RUN pip install -r requirements.txt