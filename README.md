<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/catiaspsilva/README-template">
    <img src="images/wallstreet.png" alt="Logo" width="300" height="200">
  </a>

  <p align="center">
    Financial news sentiment analysis with FinBERT applied to stock market prediction
    <br />
    <a href="https://huggingface.co/spaces/DimaTivator/stonks">View Demo</a>
  </p>
</p>


<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



## About The Project

Many research papers and the experience of firms like Bloomberg suggest a strong correlation between financial news sentiment and stock market movements. This project provides a lightweight model fine-tuned for financial sentiment analysis.

As a base model, I chose <a href="https://huggingface.co/ProsusAI/finbert">Prosus AI FinBERT</a>, a BERT-based model already fine-tuned for financial sentiment classification. I further improved its performance by scaling up the dataset size.

For stock price forecasting, I used Vector Auto-Regression (VAR) as the baseline model. It predicts future closing prices for each company based on closing prices and average news sentiment over the past 60 weekdays. The forecast horizon is set to 10 days. As more financial news data is collected, the input window size may be increased to enhance prediction accuracy.

The project includes a Streamlit demo hosted on Hugging Face. Every six hours, scrapers update the dataset, and the price forecast is recalculated.

## Getting Started

### Dependencies

The list of project requirements is fairly standard and can be found in requirements.txt. To create a new virtual environment and install all dependencies, run:
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirments.txt
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/DimaTivator/stonks.git
   ```

2. Setup the environment and install all dependencies, as shown in <a href="#Dependencies">Dependencies</a> section.

3. Run `startup.sh` to collect news data, load the model, generate forecasts, and launch the Streamlit application.


## Project Structure

`data/` - Contains collected news data, preprocessed texts, stocks prices and forecasts

`images/` – Stores images used for documentation

`research/` – Includes Jupyter notebooks for experiments, data analysis, and model evaluation

`weights/` – Directory for storing model weights

`data_pipeline.py` – Main script that manages data collection and forecasting

`app.py` – The main Streamlit application

`config.py` – Configuration file that stores constants and paths

`crontab` – Cron job configuration file for automating periodic data updates

`model.py` – Contains ML models for sentiment analysis and forecasting

`price_forecast.py` – Processes stock data and runs forecasting model

`scraper.py` – Fetches financial news and stock market data

`sentiment.py` – Processes news and computes sentiment scores for them


## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## Authors

Dmtrii Andriianov - andriianov164@gmail.com


## Acknowledgements

* [ProcusAI](https://huggingface.co/ProsusAI/finbert)
* [FNSPID](https://arxiv.org/pdf/2402.06698)

## Licence

Distributed under the MIT License. See `LICENSE` for more information.

**Disclaimer: I am sharing codes for academic purpose under the MIT education license. Nothing herein is financial advice, and NOT a recommendation to trade real money.**
