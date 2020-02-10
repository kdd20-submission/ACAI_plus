FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN pip install --upgrade pip
RUN apt-get update -y \
    && apt install python3 -y \
    && apt install python3-pip -y

COPY requirements_docker.txt /home
RUN pip3 install --no-cache-dir -r /home/requirements_docker.txt

WORKDIR /home/acaiplus
ENTRYPOINT ["/bin/bash", "/home/acaiplus/run.sh"]