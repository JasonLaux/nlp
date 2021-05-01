import json
from graph_sort import create_graph
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_TRAIN_PATH = './data/train.data.jsonl'
LABEL_TRAIN_PATH = './data/train.label.json'
DATA_DEV_PATH = './data/dev.data.jsonl'
LABEL_DEV_PATH = './data/dev.label.json'
tokenizer = Tokenizer(oov_token="<UNK>")

'''
Use Logistic Regression model 
'''

def numeric_label(label):
    if label == 'rumour':
        return 0
    else:
        return 1


def load_data(data_fn, label_fn, train_or_test):
    sentences_train = []
    y_train = []

    with open(label_fn, encoding="utf-8") as f_l:
        label_dict = json.load(f_l)

    with open(data_fn, encoding="utf-8") as f:
        str_list = f.readlines()
        for line in str_list:
            items = json.loads(line)
            id_str = items[0]["id_str"]
            idx_list = [pair[0] for pair in create_graph(items).topological_sort()]
            sentence = ""
            for idx in idx_list:
                username = items[idx]["user"]["name"]
                text = items[idx]["text"]
                sentence += username + ":" + text
            sentences_train.append(sentence)
            y_train.append(numeric_label(label_dict.get(id_str)))

        if train_or_test is True:
            tokenizer.fit_on_texts(sentences_train)
        x_train = tokenizer.texts_to_matrix(sentences_train, mode="count")  # BOW representation

        assert len(x_train) == len(y_train)

    return x_train, y_train


def main():
    x_train, y_train = load_data(DATA_TRAIN_PATH, LABEL_TRAIN_PATH, True)
    print("Finish loading training data...")
    x_dev, y_dev = load_data(DATA_DEV_PATH, LABEL_DEV_PATH, False)
    print("Finish loading dev data...")
    classifier = LogisticRegression(max_iter=1000)
    print("Start training...")
    classifier.fit(x_train, y_train)
    print("Finish training...")
    score = classifier.score(x_dev, y_dev)
    print(score)
    y_pred = classifier.predict(x_dev)
    print("f1 score: ", f1_score(y_dev, y_pred, average="macro"))
    print("precision: ", precision_score(y_dev, y_pred, average="macro"))
    print("recall: ", recall_score(y_dev, y_pred, average="macro"))
    print("accuracy: ", accuracy_score(y_dev, y_pred))


if __name__ == '__main__':
    main()
