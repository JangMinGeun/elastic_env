from PIL import Image
import torch
from torch import nn, optim
import glob
import os
import pandas as pd
import json
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from tqdm import tqdm
import random
from matplotlib.pyplot import imshow
import torchtext
import nltk, re, string, collections
from nltk.util import ngrams
import collections
import pickle
import cv2
# from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

BATCH_SIZE = 128
EPOCH = 5

TRAIN_DESCRIPT_ROOT = "results/orn/rlb_1_22_c_lp_eval/train/rlb_1_22_c_lp_eval_descript.pickle"
TRAIN_IMG_ROOT = "results/orn/rlb_1_22_c_lp_eval/train/input/"
TEST_DESCRIPT_ROOT = "results/orn/rlb_1_22_c_lp_eval/test/rlb_1_22_c_lp_eval_descript.pickle"
TEST_IMG_ROOT = "results/orn/rlb_1_22_c_lp_eval/test/input/"

COLOR = ['lime', 'blue', 'cyan', 'teal', 'navy', 'green', 'red']
SHAPE = ['circle', 'square', 'triangle', 'hexagon']
# SHAPE = ['zero', 'four', 'three', 'six']
# SHAPE = ['0', '4', '3', '6']
class MemeDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []
        for img_path, captions in tqdm(data.items()):
            self.img_paths.append(img_path)
            self.captions.append(captions)
        self.processed_cache = {}
        # for img_path in tqdm(data):
        #     image = Image.open(img_path)
        #     image = image.resize((64,64))
        #     self.processed_cache[img_path] = self.preprocess(image)
        self.img_paths_set = list(data.keys())
        self.path2label = {path: i for i, path in enumerate(self.img_paths_set)}
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return img_path, caption, label


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        # for l in self.labels_set:
        #     np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               0:self.n_samples])
                # self.used_label_indices_count[class_] += self.n_samples
                # if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                #     np.random.shuffle(self.label_to_indices[class_])
                #     self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples  #self.n_samples == 1 fixed 

    def __len__(self):
        return self.n_dataset // self.batch_size

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def sample1Caption(img_path, corpus, model, path2caption, num_cand):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    i = 0
    txts = []
    common = " and the number of vertex is " 
    # while i < 28:
    #     txt = random.choice(corpus)
        
    #     if txt in txts:
    #         continue
    #     if len(txt)>72:
    #         continue
    #     i += 1
    #     if txt in path2caption[img_path]:
    #         txts.insert(0, txt)
    #     else:
    #         txts.append(txt)
    # captions = path2caption[img_path]
    # captions = captions.replace("circle", "zero")
    # captions = captions.replace("square", "four")
    # captions = captions.replace("triangle", "three")
    # captions = captions.replace("hexagon", "six")
    for i in range (7):
        txt = COLOR[i] + common
        txt = COLOR[i] + ' '
        for j in range (4):
            full_txt = txt + SHAPE[j]
            if full_txt in path2caption[img_path]:
                txts.insert(0, full_txt)
                # print("good!!!!!!!!!!")
            else :
                txts.append(full_txt)
    # flag = 0
    # for txt in txts:
    #     if txt in path2caption[img_path]:
    #         flag = 1
    # if flag == 0: # there is no right answer
    #     pos_split = path2caption[img_path]
    #     txts.insert(0, random.choice(pos_split))
    #     print("insertion phase operated.")
    #     print(name)
    #     print(f"Positive caption: {pos_txt}")
    #     print(f"Negative caption: {neg_txts}")
    # txts.insert(0, path2caption[img_path])
    text = clip.tokenize(txts).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    #     print("Label probs:", probs)
    #     print(np.argmax(probs))
    #     imshow(np.asarray(Image.open(img_path)))
    index = np.argmax(probs)
    rank = np.argsort(probs, axis=1)
    # print("max_index:", index)
    # print("max_value:", probs[0][index])
    # print("rank_value:", probs[0][rank[0][0]])
    return txts[index], index

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

train_img_paths = glob.glob(os.path.join(TRAIN_IMG_ROOT, "*.png"))
test_img_paths = glob.glob(os.path.join(TEST_IMG_ROOT, "*.png"))

d = {}
with open (TRAIN_DESCRIPT_ROOT, "rb") as fr:
    train_des_data = pickle.load(fr)
with open (TEST_DESCRIPT_ROOT, "rb") as fr:
    test_des_data = pickle.load(fr)
    

for i, img_path in enumerate(train_img_paths):
    captions = train_des_data[0][i]
    d[img_path] = captions

for i, img_path in enumerate(test_img_paths):
    captions = test_des_data[0][i]
    d[img_path] = captions

# image = Image.open(train_img_paths[0])
# image = image.resize((64,64))
# image = np.array(image)
# print(image.shape)





d_train = {k: d[k] for k in train_img_paths}
d_test = {k: d[k] for k in test_img_paths}
# print(len(d_train))
# print(len(d_test))
train_dataset = MemeDataset(d_train, preprocess)
test_dataset = MemeDataset(d_test, preprocess)
# print(len(train_dataset))
# print(len(test_dataset)) 
# print(test_dataset[200])

train_labels = torch.tensor([item[2] for item in train_dataset])
train_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE, 1)
train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

test_labels = torch.tensor([item[2] for item in test_dataset])
test_sampler = BalancedBatchSampler(test_labels, BATCH_SIZE, 1)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)


loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCH)

best_te_loss = 1e5
best_ep = -1


# torch.save(model.state_dict(), "last_model.pt")
# model evaluation

# model.load_state_dict(torch.load("best_model_shape.pt")) # best model load 
# model.load_state_dict(torch.load("best_model_number.pt")) # best model load 


NUM_NEG = 14
NUM_TEST = 1000

n_correct = 0
for i in tqdm(range(NUM_TEST)):
    empty = True
    while empty:
        img_path = random.choice(list(d_test.keys()))
        img = Image.open(img_path)
        img = img.resize((64,64))
        image = preprocess(img).unsqueeze(0).to(device)
        name = img_path.split('/')[-1].split('.')[0]
        caps = d_test[img_path]
        if len(caps) > 0:
            # print('len_caps:', len(caps))
            # caps_split = caps.split()
            # pos_txt = random.choice(caps_split)
            pos_txt = caps
            # print(f"Positive caption: {pos_txt}")
        #         pos_txt = ' '.join(pos_txt)
            empty = False
#     print(pos_txt)
    neg_i = 0
    neg_txts = []
    while neg_i < NUM_NEG:
        img_path = random.choice(list(d_test.keys()))
        neg_name = img_path.split('/')[-1].split('.')[0]
        if neg_name == name:
            continue
        caps = d_test[img_path]
        if len(caps) == 0:
            continue
        # caps_split = caps.split()
        neg_txt = caps
        if neg_txt in neg_txts:
            continue
        neg_txts.append(neg_txt)
        neg_i += 1
    # print(name)
    # print(f"Positive caption: {pos_txt}")
    # print(f"Negative caption: {neg_txts}")
    text = clip.tokenize([pos_txt]+neg_txts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#     print("Label probs:", probs)
#     print(np.argmax(probs))
    if np.argmax(probs) == 0:
        n_correct +=1
print("probs_shape:", probs.shape)
print(f"Test precision {n_correct/NUM_TEST}")
img = Image.open(img_path)
img = img.resize((64,64))
img.save('example.jpg',"JPEG")


corpus = []
for txtlist in d_train.values():
    # txt_split = txtlist.split()
    # corpus += txt_split
    corpus.append(txtlist)
print('len_corpus:', len(corpus))
print('corpus[0]:', corpus[0])
captions = {}
for img_path in tqdm(d_test.keys()):
    caption, index = sample1Caption(img_path, corpus, model, d, 100)
    # if rank < 100:
    #     print('rank :', rank)
    #     print('not bad!!')
    captions[img_path] = caption

for get_bleu in range(1,4):
    bleu_x_lst = []
    bleu_y_lst = []
    for p, caps in d_test.items():
        if not caps:
            continue
        bleu_x_lst.append(captions[p].split())
        splittedcaps = [x.split() for x in caps]
        bleu_y_lst.append(splittedcaps)
    BLEU = torchtext.data.metrics.bleu_score(bleu_x_lst, bleu_y_lst, max_n=get_bleu, weights=[1/get_bleu]*get_bleu)
    print(f"{get_bleu}-gram BLEU score: {BLEU}")

sentences = list(captions.values())
BigramCtr = collections.Counter()
UnigramCtr = collections.Counter()
for sentence in sentences:
    BigramCtr.update(nltk.ngrams(sentence, 2))
    UnigramCtr.update(nltk.ngrams(sentence, 1))
# print("Unigram count:",len(BigramCtr)/len(sentences))
# print("Bigram count:",len(UnigramCtr)/len(sentences))
print("Unigram count:",len(BigramCtr))
print("Bigram count:",len(UnigramCtr))

# num_circle_correct = 0
# num_square_correct = 0
# num_hexa_correct = 0

# total_circle = 0
# total_square = 0
# total_hexa = 0
# for i in tqdm(range(3000)):
#     seen_path = random.choice(list(d_train.keys()))
#     pred_cap_seen = sample1Caption(seen_path, corpus, model, d, 100)
#     gt_cap_seen = d_train[seen_path]
#     if "circle" in gt_cap_seen:
#         total_circle += 1
#     elif "square" in gt_cap_seen:
#         total_square += 1
#     elif "hexagon" in gt_cap_seen:
#         total_hexa += 1
#     # imshow(Image.open(seen_path))
#     if pred_cap_seen[0] in gt_cap_seen:
#         if "circle" in gt_cap_seen:
#             num_circle_correct += 1
#         elif "square" in gt_cap_seen:
#             num_square_correct += 1
#         elif "hexagon" in gt_cap_seen:
#             num_hexa_correct += 1
# # print(f"Some ground truth captions for this seen image: {gt_cap_seen}")
# # print(f"Caption sampled by fintuned CLIP for this seen image: {pred_cap_seen}")
# print("total seen circle precision : ", num_circle_correct / total_circle)
# print("total seen square precision : ", num_square_correct / total_square)
# print("total seen hexa precision : ", num_hexa_correct / total_hexa)

num_circle_correct = 0
num_square_correct = 0
num_hexa_correct = 0
num_tri_correct = 0

total_circle = 0
total_square = 0
total_hexa = 0
total_tri = 0

for i in tqdm(range(4000)):
    unseen_path = list(d_test.keys())[i]
    pred_cap_unseen = sample1Caption(unseen_path, corpus, model, d, 100)
    # imshow(Image.open(unseen_path))
    gt_cap_unseen = d_test[unseen_path]
    if "circle" in gt_cap_unseen:
        total_circle += 1
    elif "square" in gt_cap_unseen:
        total_square += 1
    elif "hexagon" in gt_cap_unseen:
        total_hexa += 1
    else:
        total_tri += 1    
    # captions = gt_cap_unseen    
    # captions = captions.replace("circle", "zero")
    # captions = captions.replace("square", "four")
    # captions = captions.replace("triangle", "three")
    # captions = captions.replace("hexagon", "six")
    
    if pred_cap_unseen[0] == gt_cap_unseen:
        if "circle" in gt_cap_unseen:
            num_circle_correct += 1
        elif "square" in gt_cap_unseen:
            num_square_correct += 1
        elif "hexagon" in gt_cap_unseen:
            num_hexa_correct += 1
        else:
            num_tri_correct += 1
    # print(f"Some ground truth captions for this unseen image: {gt_cap_unseen}")
    # print(f"Caption sampled by fintuned CLIP for this unseen image: {pred_cap_unseen}")
print("total unseen circle precision : ", num_circle_correct / total_circle)
print("total unseen square precision : ", num_square_correct / total_square)
print("total unseen hexa precision : ", num_hexa_correct / total_hexa, total_hexa)
print("total unseen tri precision : ", num_tri_correct / total_tri)

