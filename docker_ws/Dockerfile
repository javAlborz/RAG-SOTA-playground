FROM pgvector/pgvector:0.7.4-pg17

RUN apt update && apt upgrade -y
RUN apt install golang-go -y
RUN apt install ca-certificates -y
RUN apt install curl -y
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt install python3-psycopg2 -y


COPY ./requirements.txt /home/
WORKDIR /home/
RUN pip3 install -r requirements.txt --break-system-packages

WORKDIR /app/pgvector-rag-app/

# create unprivileged user
# RUN useradd -s /bin/bash noob_user
# RUN mkdir -p /home/noob_user
# RUN chown -R noob_user:noob_user /home/noob_user
# USER noob_user
