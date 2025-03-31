#!/bin/bash
python data_pipeline.py &

while true; do
  sleep 6h
  python data_pipeline.py
done &

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
