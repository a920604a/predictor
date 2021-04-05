FROM pytorch/pytorch:1.6
LABEL maintainer="a920604a@gmail.com"

RUN set -x \
    && apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y libmysqlclient-dev

RUN apt update
RUN apt install -y vim 

RUN set -x \
	&& apt-get update \
	&& apt-get install --no-install-recommends --no-install-suggests -y vim libsm6 libxrender1 libxext6 libgl1-mesa-glx libglib2.0-dev mysql-server \
    && pip install numpy XlsxWriter sklearn opencv-python SQLAlchemy mysqlclient autopep8 PyYAML loguru pymysql\
                   scalene 

RUN mkdir /logs && chmod 777 -R /logs/
COPY ./app/ /app/
WORKDIR /app
