FROM continuumio/miniconda3

WORKDIR /workspace
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["bash", "-lc"]
RUN echo "conda activate solar-power" >> ~/.bashrc
ENV PATH /opt/conda/envs/solar-power/bin:$PATH

COPY . /workspace
CMD ["bash"]
