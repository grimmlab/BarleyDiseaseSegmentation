FROM nvcr.io/nvidia/pytorch:23.09-py3
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y htop
RUN apt-get install -y nano
# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1-mesa-glx -y
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

