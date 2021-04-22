import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
import json
from graph_sort import create_graph

DATA_TRAIN_PATH = './data/train.data.jsonl'
LABEL_TRAIN_PATH = './data/train.label.json'
DATA_DEV_PATH = './data/dev.data.jsonl'
LABEL_DEV_PATH = './data/dev.label.json'
DATA_TEST_PATH = './data/test.data.jsonl'

'''
output: tokens_ids_tensor, attn_mask_tensor, seg_ids_tensor, label
'''


class TweetDataset(Dataset):

    def __init__(self, fn_data, fn_label=None, maxlen=256):
        # Store the contents of the file in a pandas dataframe
        self.data = open(fn_data, encoding="utf-8").readlines()
        if fn_label is not None:
            self.label_dict = json.load(open(fn_label, encoding="utf-8"))
        else:
            self.label_dict = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen  # the max length of the sentence in the corpus

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items = json.loads(self.data[index])
        id_str = items[0]["id_str"]  # Tweet index
        idx_list = [pair[0] for pair in create_graph(items).topological_sort()]
        tokens_list = []
        for idx in idx_list:
            username = items[idx]["user"]["name"]
            text = items[idx]["text"]
            current_sentence = username + ":" + text
            tokens_list.append(self.tokenizer.tokenize(current_sentence))

        tokens_concat = ['[CLS]'] + [token for item in tokens_list for token in item] + ['[SEP]']
        if len(tokens_concat) < self.maxlen:
            padded_tokens = tokens_concat + ['[PAD]' for _ in range(self.maxlen - len(tokens_concat))]
        else:
            padded_tokens = tokens_concat[:self.maxlen - 1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)
        attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_masks_tensor = torch.tensor(attn_mask)
        label_idx = torch.tensor(-1)
        if self.label_dict:
            label = self.label_dict[id_str]  # Tweet label
            if label == "non-rumour":
                label_idx = torch.tensor(1)
            else:
                label_idx = torch.tensor(0)
        return tokens_ids_tensor, attn_masks_tensor, label_idx, id_str


class RumourClassifier(nn.Module):

    def __init__(self):
        super(RumourClassifier, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768
        # output dimension is 1 because we're working with a binary classification problem
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, tokens_ids, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(input_ids=tokens_ids, attention_mask=attn_masks)
        cont_reps = outputs.last_hidden_state

        # Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def evaluate(net, criterion, dataloader, gpu):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels, _ in dataloader:
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    if count == 0:
        raise KeyError("Dataloader is loaded incorrectly!")
    else:
        return mean_acc / count, mean_loss / count


def train(net, criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_acc = 0
    st = time.time()
    for ep in range(max_eps):

        for it, (seq, attn_masks, labels, _) in enumerate(train_loader):
            # Clear gradients
            opti.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks)

            # Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            opti.step()

            if it % 100 == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}"
                      .format(it, ep, loss.item(), acc, (time.time() - st)))
                st = time.time()

        dev_acc, dev_loss = evaluate(net, criterion, dev_loader, gpu)
        print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}".format(ep, dev_acc, dev_loss))
        if dev_acc > best_acc:
            print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, dev_acc))
            best_acc = dev_acc
            torch.save(net.state_dict(), 'sstcls_{}.pth'.format(ep))
    return net


def predict(net, dataloader, gpu):
    net.eval()
    dict_pred = {}
    with torch.no_grad():
        for seq, attn_masks, _, id_str in dataloader:
            seq, attn_masks = seq.cuda(gpu), attn_masks.cuda(gpu)
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
    return dict_pred

def main():
    gpu = 0
    print("Creating the classifier, initialised with pretrained BERT-BASE parameters...")
    net = RumourClassifier()
    net.cuda(gpu)  # Enable gpu support for the model
    print("Done creating the classifier.")
    # Define loss function based on binary cross-entropy.
    criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(net.parameters(), lr=2e-5)
    num_epoch = 10

    train_dataset = TweetDataset(DATA_TRAIN_PATH, LABEL_TRAIN_PATH, maxlen=256)
    dev_dataset = TweetDataset(DATA_DEV_PATH, LABEL_DEV_PATH, maxlen=256)
    train_dataloader = DataLoader(train_dataset, batch_size=10)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10)
    test_dataset = TweetDataset(DATA_DEV_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=10)

    net_trained = train(net, criterion, opti, train_dataloader, dev_dataloader, num_epoch, gpu)


if __name__ == '__main__':
    main()
