#FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel 
FROM xodls4179/aetin:hmd_sub
MAINTAINER xodls4179@cau.ac.kr

#ENV LC_ALL=C.UTF-8 

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

    # Install Python Packages 
RUN pip install datasets>=1.18.3 && \
    pip install transformers==4.11.3 && \
    pip install jiwer



