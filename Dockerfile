FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

COPY --chown=user weights /app/weights
COPY --chown=user *.py /app/

COPY --chown=user startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

EXPOSE 8501

CMD ["/app/startup.sh"]
