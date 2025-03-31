#!/bin/bash
python data_pipeline.py &

service cron start

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
