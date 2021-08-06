# Based on https://github.com/cb372/docker-sample/blob/master/Dockerfile

# Using Centos 8 As Base Image
FROM centos:centos8

# Installing Required Tools
RUN yum -y --noplugins --verbose install git wget tar

RUN git clone https://github.com/sraoss/pgsql-ivm
RUN cd pgsql-ivm && git checkout 3bf6953688153fa72dd48478a77e37cf3111a1ee && cd ..


RUN yum -y install gcc
RUN yum install -y bison readline zlib openssl
RUN yum groupinstall -y 'Development Tools'

RUN yum install -y flex readline-devel
RUN yum install -y zlib-devel

# Installing PIP
RUN  yum install -y python2
RUN wget "https://bootstrap.pypa.io/pip/2.7/get-pip.py"  && python2 get-pip.py
RUN ln -s /usr/bin/python2 /usr/bin/python
RUN yum install -y python2-devel

RUN cd pgsql-ivm && ./configure --with-python  && make && make install
RUN adduser postgres
RUN mkdir /usr/local/pgsql/data
RUN chown postgres /usr/local/pgsql/data
USER postgres
RUN  /usr/local/pgsql/bin/initdb -D /usr/local/pgsql/data 
RUN  /usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data/ start &&  /usr/local/pgsql/bin/createdb test
USER root


RUN yum install -y m4 flex bison graphviz 

RUN wget "https://cmake.org/files/v3.11/cmake-3.11.0.tar.gz" \
&& tar xzf  cmake-3.11.0.tar.gz \
&& rm cmake-3.11.0.tar.gz \
&& cd cmake-3.11.0 \
&& ./bootstrap --prefix=/usr/local/\
&& make \
&& make install \
&& cd ..


# Addingtdb Code Directory

RUN mkdir /usr/local/tdb
RUN mkdir /usr/local/tdb/build

ADD ./cmake /usr/local/tdb/cmake
ADD ./deploy /usr/local/tdb/deploy
ADD ./doc /usr/local/tdb/doc
ADD ./licenses /usr/local/tdb/licenses
ADD ./methods /usr/local/tdb/methods

ADD ./CMakeLists.txt /usr/local/tdb/CMakeLists.txt
ADD ./configure /usr/local/tdb/configure
ADD ./LICENSE /usr/local/tdb/LICENSE
ADD ./Makefile /usr/local/tdb/Makefile

ADD ./scripts /usr/local/tdb/scripts
ADD ./src /usr/local/tdb/src

RUN cd /usr/local/tdb/ && ./configure -DPOSTGRESQL_13_PG_CONFIG=/usr/local/pgsql/bin/pg_config 
RUN	cd /usr/local/tdb/build && make  -j 1; exit 0 
RUN cd /usr/local/tdb/build && make  -j 1 

RUN yum install -y passwd which
RUN echo "postgres" | passwd --stdin postgres

RUN runuser -l postgres -c '/usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data/ start'  \
            && cd /usr/local/tdb/build \
            && export PATH=$PATH:/usr/local/pgsql/bin \
            && src/bin/madpack -s madlib -p postgres -c postgres/postgres@localhost:5432/test install


# Setting the timezone
ENV TZ=America/Los_Angeles 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ADD ./ui /usr/local/tdb/ui


RUN python -m pip install flask flask_cors
RUN export PATH=$PATH:/usr/local/pgsql/bin && python -m pip install psycopg2 psycopg2-binary
RUN echo "export LD_LIBRARY_PATH=/usr/local/pgsql/lib" >> ~/.bashrc

ADD ./backend /usr/local/tdb/backend
ADD ./data /usr/local/tdb/data

USER postgres
RUN  /usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data/ start &&  /usr/local/pgsql/bin/psql test < /usr/local/tdb/data/sql/metadata_tables.sql \
    &&  /usr/local/pgsql/bin/psql test < /usr/local/tdb/data/sql/tweets.sql \
    &&  /usr/local/pgsql/bin/psql test < /usr/local/tdb/data/sql/tweets_full.sql \    
    &&  /usr/local/pgsql/bin/psql test < /usr/local/tdb/data/sql/tweets_state.sql 

USER root

RUN python -m pip install sqlparse sklearn requests

# Command To Run The Server (sleep 30 provided to give MySQL time to start)
CMD runuser -l postgres -c '/usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data/ start'  \
            && cd /usr/local/tdb/backend/flaskr && python __init__.py

# Forwarded HTTP ports
EXPOSE 5432
EXPOSE 8000
EXPOSE 80
EXPOSE 8080
EXPOSE 5000
