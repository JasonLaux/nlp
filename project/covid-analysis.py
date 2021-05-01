from bert_detection import RumourClassifier
import torch
import re
import random
from torch.utils.data import DataLoader
import json
from bert_detection import TweetDataset
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np

# Trained model from Task 1. Specify the path...
MODEL_PATH = "sstcls.pth"

# Relevant data path
COVID_DATA_PATH = "./data/covid.data.jsonl"
COVID_LABEL_PATH = './data/covid_label.json'
DEV_DATA_PATH = './data/dev.data.jsonl'
DEV_LABEL_PATH = './data/dev.label.json'
TRAIN_DATA_PATH = './data/train.data.jsonl'
TRAIN_LABLE_PATH = './data/train.label.json'
TEST_DATA_PATH = './data/test.data.jsonl'


def load_model():
    device = torch.device('cpu')
    model = RumourClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\/\/\S+)", " ", tweet).split())


def sa_text(score):
    if score > 0:
        output = "Positive"
    elif score < 0:
        output = "Negative"
    else:
        output = "Neutral"
    return output


def predict(net, dataloader):
    net.eval()
    dict_pred = {}
    with torch.no_grad():
        for seq, attn_masks, _, id_str in dataloader:
            logits = net(seq, attn_masks)
            probs = torch.sigmoid(logits.unsqueeze(-1))
            y_pred = (probs > 0.5).long().squeeze()
            if y_pred == torch.tensor(0):
                label = "rumour"
            else:
                label = "non-rumour"
            dict_pred.update({id_str[0]: label})
    with open("./data/test_pred_label.json", "w+") as f:
        json.dump(dict_pred, f)
    # return dict_pred


def classify_dataset():
    non_rumours_datalist = []
    rumours_datalist = []

    with open(COVID_LABEL_PATH, encoding="utf-8") as label_reader:

        labels = [item[1] for item in json.load(label_reader).items()]

    with open(COVID_DATA_PATH, encoding="utf-8") as f:
        data_obj = f.readlines()
        for idx, label in enumerate(labels):
            if label == 'non-rumour':
                non_rumours_datalist.append(json.loads(data_obj[idx]))
            else:
                rumours_datalist.append(json.loads(data_obj[idx]))

    return rumours_datalist, non_rumours_datalist


def sa_score_replies(rumours_datalist, non_rumours_datalist):
    rumour_texts = []
    non_rumour_texts = []
    rumours_sa = []
    non_rumours_sa = []
    print("rumours_datalist...")
    for item in rumours_datalist:
        rumour_text = ""
        for tweet in item:
            rumour_text += (tweet["text"] + " ")
        rumour_texts.append(rumour_text)
    print("non_rumours_datalist...")
    for item in non_rumours_datalist:
        non_rumour_text = ""
        for tweet in item:
            non_rumour_text += (tweet["text"] + " ")
        non_rumour_texts.append(non_rumour_text)
    print("rumour_texts...")
    for text in rumour_texts:
        text = clean_tweet(text)
        polarity = TextBlob(text).sentiment[0]
        rumours_sa.append([polarity, TextBlob(text).sentiment[1],
                           sa_text(polarity)])
    print("non_rumour_texts...")
    for text in non_rumour_texts:
        text = clean_tweet(text)
        polarity = TextBlob(text).sentiment[0]
        non_rumours_sa.append([polarity, TextBlob(text).sentiment[1],
                               sa_text(polarity)])

    # print(len(rumours_sa))
    # print(len(non_rumours_sa))

    rumours_sa = random.sample(rumours_sa, 1000)
    non_rumours_sa = random.sample(non_rumours_sa, 1000)

    rumours_df = pd.DataFrame(data=rumours_sa, columns=['Polarity', 'Subjectivity', 'Description'])

    non_rumours_df = pd.DataFrame(data=non_rumours_sa, columns=['Polarity', 'Subjectivity', 'Description'])

    return rumours_df, non_rumours_df


def plot_sa(rumours_df, non_rumours_df):
    labels = ['Negative', 'Neutral', 'Positive']

    rumour_counts = rumours_df.groupby(['Description']).size().reset_index(name='Counts').iloc[:, 1].tolist()
    non_rumour_counts = non_rumours_df.groupby(['Description']).size().reset_index(name='Counts').iloc[:, 1].tolist()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, rumour_counts, width, label='Rumour')
    rects2 = ax.bar(x + width / 2, non_rumour_counts, width, label='Non-rumour')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Counts')
    ax.set_title('Sentiment Polarity Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def extract_hashtag(rumour_list, non_rumour_list):
    rumour_hashtags = {}
    non_rumour_hashtags = {}
    # [{'text': 'Houston', 'indices': [18, 26]}, {'text': 'COVID19', 'indices': [27, 35]}]

    for item in rumour_list:
        hashtags = item[0]["entities"]["hashtags"]
        if hashtags:
            for obj in hashtags:
                hashtag = obj["text"].lower()
                if hashtag == 'covidー19' or hashtag == 'covid_19' or hashtag == 'covid':
                    hashtag = 'covid19'
                if hashtag in rumour_hashtags.keys():
                    v_pre = rumour_hashtags.get(hashtag)
                    rumour_hashtags.update({hashtag: v_pre + 1})
                else:
                    rumour_hashtags.update({hashtag: 1})

    for item in non_rumour_list:
        hashtags = item[0]["entities"]["hashtags"]
        if hashtags:
            for obj in hashtags:
                hashtag = obj["text"].lower()
                if hashtag == 'covidー19' or hashtag == 'covid_19' or hashtag == 'covid':
                    hashtag = 'covid19'
                if hashtag in non_rumour_hashtags.keys():
                    v_pre = non_rumour_hashtags.get(hashtag)
                    non_rumour_hashtags.update({hashtag: v_pre + 1})
                else:
                    non_rumour_hashtags.update({hashtag: 1})

    return rumour_hashtags, non_rumour_hashtags


def main():
    # # Classify the covid-19 dataset and save the predict label file as test_pred_label.json
    # dataset = TweetDataset(TEST_DATA_PATH)
    # data_loader = DataLoader(dataset, batch_size=1)
    # model = load_model()
    # predict(model, data_loader)

    # hashtags extraction
    rumours_datalist, non_rumours_datalist = classify_dataset()
    rumour_hashtags, non_rumour_hashtags = extract_hashtag(rumours_datalist, non_rumours_datalist)

    rumour_hashtags_sorted = sorted(rumour_hashtags.items(), key=lambda item: item[1], reverse=True)
    non_rumour_hashtags_sorted = sorted(non_rumour_hashtags.items(), key=lambda item: item[1], reverse=True)
    #
    print(rumour_hashtags_sorted[:10])
    print(non_rumour_hashtags_sorted[:10])

    # # sentiment scores calculation and display the result
    # rumours_datalist, non_rumours_datalist = classify_dataset()
    # rumours_df, non_rumours_df = sa_score_replies(rumours_datalist, non_rumours_datalist)
    # plot_sa(rumours_df, non_rumours_df)


if __name__ == '__main__':
    main()
