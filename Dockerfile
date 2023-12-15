FROM continuumio/miniconda3

EXPOSE 5239

WORKDIR /app

COPY . .

RUN conda env create -f env.yml
# RUN conda activate nisqa

# SHELL ["conda", "run", "-n", "nisqa", "/bin/bash", "-c"]
# RUN echo "source activate nisqa" > ~/.bashrc
# ENV PATH /opt/conda/envs/nisqa/bin:$PATH