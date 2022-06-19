FROM python:3.7
WORKDIR /
COPY app/requirements.txt /app/requirements.txt
RUN pip install -r  /app/requirements.txt

COPY ./app /app/
COPY ./app/start.sh /start.sh
RUN chmod +x /start.sh

WORKDIR /

ENV PYTHONPATH=/

CMD ["/start.sh"]
