FROM tacc/tacc-base:rockylinux8-impi19.0.9-common
RUN sed -i 's/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019/GPG-PUB-KEY-INTEL-SW-PRODUCTS/g' /etc/yum.repos.d/intelproducts.repo && dnf config-manager --set-enabled powertools && dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && dnf install -y python3.12 python3.12-pip git clang gmp-devel cmake libsodium-devel boost1.78-devel openssl-devel && alternatives --set python3 /usr/bin/python3.12 && alternatives --set python /usr/bin/python3.12 && git clone https://github.com/data61/MP-SPDZ.git /usr/src/MP-SPDZ
RUN ln --symbolic /usr/include/boost1.78/boost /usr/include/boost && ln --symbolic /usr/lib64/boost1.78/* /usr/lib64
WORKDIR /usr/src/MP-SPDZ
RUN export CC=clang && export CXX=clang++ && make setup && make -j8 libote && make -j8 replicated-ring-party.x sy-rep-ring-party.x rep4-ring-party.x semi2k-party.x && Scripts/setup-ssl.sh 4

