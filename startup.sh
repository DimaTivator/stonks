#!/bin/bash
python data_pipeline.py

python3 scheduler.py &

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
