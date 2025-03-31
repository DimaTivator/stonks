FROM python:3.9

RUN apt-get update && apt-get install -y cron

COPY crontab /etc/cron.d/data_pipeline_cron
RUN chmod 0644 /etc/cron.d/data_pipeline_cron && crontab /etc/cron.d/data_pipeline_cron

RUN touch /var/log/cron.log

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

COPY --chown=user weights /app/weights
COPY --chown=user *.py /app/

CMD ["/app/startup.sh"]
