from model import get_model
from config import Config

import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


device = Config.DEVICE


def clean_text(text):
    text = str(text)
    # remove html tags
    text = re.sub(r"<.*?>", "", text)
    # remove links
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub("/", "", text)
    # replace spaces sequences with 1 space
    text = re.sub(r"\s+", " ", text).strip()
    # replace ... with .
    text = re.sub(r"\.+", ".", text)

    return text


class TextsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        padded_labels = torch.stack(labels, dim=0)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels,
        }


def get_predictions(model, dataloader):
    model.eval()
    with torch.no_grad():
        preds = []
        for batch in tqdm(dataloader):
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = (
                model(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
            )
            probs = F.softmax(outputs, dim=-1)

            preds.append(probs)

        return torch.cat(preds, dim=0)


def aggregate_sentiment(preds):
    weights = np.array([-1, -0.5, 0, 0.5, 1])
    scores = preds @ weights
    return scores


def update_sentiments(
    tickers: list[str], data_directory: str, model_weights_path: str, model_name: str
) -> None:
    model, tokenizer = get_model(os.path.join(model_weights_path, f"{model_name}.pth"))

    for ticker in tqdm(tickers):
        file_path = os.path.join(data_directory, f"{ticker}.csv")
        if not os.path.exists(file_path):
            continue

        df_ticker = pd.read_csv(file_path)
        df = df_ticker[df_ticker["sentiment"].isna()]
        if df.empty:
            continue

        df["title"] = df["title"].apply(clean_text)

        dummy_labels = np.zeros(len(df))
        texts_dataset = TextsDataset(
            texts=df["title"], labels=dummy_labels, tokenizer=tokenizer
        )

        batch_size = 32

        texts_dataloader = DataLoader(
            texts_dataset,
            batch_size=batch_size,
            collate_fn=texts_dataset.collate_fn,
            shuffle=False,
            drop_last=False,
        )

        preds = get_predictions(model, texts_dataloader).tolist()

        sentiment_scores = [aggregate_sentiment(np.array(x)) for x in preds]
        df["sentiment"] = sentiment_scores

        df_ticker.loc[df.index, "sentiment"] = df["sentiment"]
        df_ticker.to_csv(file_path, index=False)


def update_daily_sentiments(tickers: list[str], news_path: str, sentiments_path: str):
    for ticker in tickers:
        df = pd.read_csv(os.path.join(news_path, f"{ticker}.csv"))
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["sentiment"] = df["sentiment"].astype(float)
        df = df.dropna(subset=["sentiment"])
        df = df.groupby("date")["sentiment"].mean().reset_index()

        if os.path.exists(os.path.join(sentiments_path, f"{ticker}.csv")):
            df_existing = pd.read_csv(os.path.join(sentiments_path, f"{ticker}.csv"))
            df_existing["date"] = pd.to_datetime(
                df_existing["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            df_existing["date"] = pd.to_datetime(df_existing["date"], format="%Y-%m-%d")
            df_existing = df_existing[
                df_existing["date"] < pd.to_datetime(df["date"].min())
            ]
            df = pd.concat([df_existing, df], ignore_index=True)
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
            df = df.drop_duplicates(subset=["date"], keep="last")

        df.to_csv(os.path.join(sentiments_path, f"{ticker}.csv"), index=False)


def fillna_in_daily_sentiments(tickers: list[str], sentiments_path: str):
    all_data = {}

    for ticker in tickers:
        file_path = os.path.join(sentiments_path, f"{ticker}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates(subset=["date"])
            all_data[ticker] = df

    all_dates = sorted(
        set().union(
            *[df["date"].dropna().astype(str).unique() for df in all_data.values()]
        )
    )

    for ticker, df in all_data.items():
        df = df.set_index("date").reindex(all_dates).reset_index()
        df.rename(columns={"index": "date"}, inplace=True)
        all_data[ticker] = df

    date_means = {}
    for date in all_dates:
        sentiments_sum = 0
        count_notnan = 0
        for ticker in tickers:
            score = all_data[ticker][all_data[ticker]["date"] == date][
                "sentiment"
            ].mean()
            if date in all_data[ticker]["date"].values and not np.isnan(score):
                sentiments_sum += score
                count_notnan += 1
        date_means[date] = sentiments_sum / count_notnan

    for ticker in tickers:
        mask = all_data[ticker]["sentiment"].isna()
        all_data[ticker].loc[mask, "sentiment"] = (
            all_data[ticker].loc[mask, "date"].map(date_means)
        )

    for ticker, df in all_data.items():
        df.to_csv(os.path.join(sentiments_path, f"{ticker}.csv"), index=False)
