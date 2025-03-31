#!/bin/bash
python data_pipeline.py

service cron start

streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true \
    --server.fileWatcherType none