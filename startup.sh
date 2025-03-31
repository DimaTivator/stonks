#!/bin/bash
python data_pipeline.py

python3 scheduler.py &

streamlit run app.py \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.fileWatcherType none
