FROM tacc/tacc-base:rockylinux8-impi19.0.9-common
RUN sed -i 's/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019/GPG-PUB-KEY-INTEL-SW-PRODUCTS/g' /etc/yum.repos.d/intelproducts.repo
RUN dnf install -y python3.12 python3.12-pip git && alternatives --set python3 /usr/bin/python3.12 && alternatives --set python /usr/bin/python3.12
RUN git clone https://github.com/shaneMenzies/private-pgm.git /tmp/private-pgm && git clone https://github.com/MarieOestreich/PRO-GENE-GEN.git /tmp/PRO-GENE-GEN && git clone https://github.com/shaneMenzies/ppml_tools.git /tmp/ppml_tools
RUN pip3.12 install git+https://github.com/shaneMenzies/private-pgm.git
RUN pip3.12 install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3.12 install -r /tmp/PRO-GENE-GEN/requirements.txt && pip3.12 install -r /tmp/ppml_tools/requirements.txt
