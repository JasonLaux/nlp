from bert_detection import RumourClassifier
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
import json
from graph_sort import create_graph
from bert_detection import TweetDataset
import pandas as pd
from textblob import TextBlob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "sstcls.pth"
COVID_DATA_PATH = "./data/covid.data.jsonl"
COVID_LABEL_PATH = './data/covid_label.json'
DEV_DATA_PATH = './data/dev.data.jsonl'


def load_model():
    device = torch.device('cpu')
    model = RumourClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def classify_tweets(model, data_loader):
    pass


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
    with open("project/data/test_label.json", "w+") as f:
        json.dump(dict_pred, f)
    # return dict_pred


# def sa_score():
#     rumours_datalist = []
#     non_rumours_datalist = []
#     labels = [item[1] for item in json.load(open('./data/covid_label.json', encoding="utf-8")).items()]
#     with open('./data/covid.data.jsonl', encoding="utf-8") as f:
#         data_obj = f.readlines()
#         for idx, label in enumerate(labels):
#             if label == 'non-rumour':
#                 non_rumours_datalist.append(json.loads(data_obj[idx])[0])
#             else:
#                 rumours_datalist.append(json.loads(data_obj[idx])[0])
#
#     rumours_sa = [[TextBlob(tweet["text"]).sentiment[0], TextBlob(tweet["text"]).sentiment[1],
#                    sa_text(TextBlob(tweet["text"]).sentiment[0])] for tweet in rumours_datalist]
#
#     non_rumours_sa = [[TextBlob(tweet["text"]).sentiment[0], TextBlob(tweet["text"]).sentiment[1],
#                        sa_text(TextBlob(tweet["text"]).sentiment[0])] for tweet in non_rumours_datalist]
#
#     # rumours_sa = rumours_sa[:4001]
#     # non_rumours_sa = non_rumours_sa[:4001]
#
#     rumours_df = pd.DataFrame(data=rumours_sa, columns=['Polarity', 'Subjectivity', 'Description'])
#
#     non_rumours_df = pd.DataFrame(data=non_rumours_sa, columns=['Polarity', 'Subjectivity', 'Description'])
#
#     print(rumours_df.groupby(['Description']).size().reset_index(name='counts'))
#     print(non_rumours_df.groupby(['Description']).size().reset_index(name='counts'))


def classify_dataset():
    non_rumours_datalist = []
    rumours_datalist = []

    with open('./data/covid_label.json', encoding="utf-8") as label_reader:

        labels = [item[1] for item in json.load(label_reader).items()]

    with open('./data/covid.data.jsonl', encoding="utf-8") as f:
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

    for item in rumours_datalist:
        rumour_text = ""
        for tweet in item:
            rumour_text += (tweet["text"] + " ")
        rumour_texts.append(rumour_text)

    for item in non_rumours_datalist:
        non_rumour_text = ""
        for tweet in item:
            non_rumour_text += (tweet["text"] + " ")
        non_rumour_texts.append(non_rumour_text)

    for text in rumour_texts:
        rumours_sa.append([TextBlob(text).sentiment[0], TextBlob(text).sentiment[1],
                           sa_text(TextBlob(text).sentiment[0])])

    for text in non_rumour_texts:
        non_rumours_sa.append([TextBlob(text).sentiment[0], TextBlob(text).sentiment[1],
                               sa_text(TextBlob(text).sentiment[0])])

    rumours_sa = rumours_sa[:4000]
    non_rumours_sa = non_rumours_sa[:4000]

    rumours_df = pd.DataFrame(data=rumours_sa, columns=['Polarity', 'Subjectivity', 'Description'])

    non_rumours_df = pd.DataFrame(data=non_rumours_sa, columns=['Polarity', 'Subjectivity', 'Description'])

    return rumours_df.groupby(['Description']).size().reset_index(name='Counts'), non_rumours_df.groupby(
        ['Description']).size().reset_index(name='Counts')


def plot_sa(rumour, non_rumour):
    labels = ['Negative', 'Neutral', 'Positive']
    rumour_counts = rumour.iloc[:, 1].tolist()
    non_rumour_counts = non_rumour.iloc[:, 1].tolist()

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
    # dataset = TweetDataset(DEV_DATA_PATH)
    # data_loader = DataLoader(dataset, batch_size=1)
    # model = load_model()
    # predict(model, data_loader)
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    rumours_datalist, non_rumours_datalist = classify_dataset()
    # sa_score_replies()

    # # hashtags extraction
    # rumours_datalist, non_rumours_datalist = classify_dataset()
    # rumour_hashtags, non_rumour_hashtags = extract_hashtag(rumours_datalist, non_rumours_datalist)
    #
    # rumour_hashtags_sorted = sorted(rumour_hashtags.items(), key=lambda item: item[1], reverse=True)
    # non_rumour_hashtags_sorted = sorted(non_rumour_hashtags.items(), key=lambda item: item[1], reverse=True)
    #
    # print(rumour_hashtags_sorted[:10])
    # print(non_rumour_hashtags_sorted[:10])

    # sentiment scores calculation and display the result
    rumours_df, non_rumours_df = sa_score_replies(rumours_datalist, non_rumours_datalist)
    plot_sa(rumours_df, non_rumours_df)


if __name__ == '__main__':
    main()
