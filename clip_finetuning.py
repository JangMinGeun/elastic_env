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

TRAIN_DESCRIPT_ROOT = "results/orn/rlb_1_22_c_lp/train/rlb_1_22_c_lp_descript.pickle"
TRAIN_IMG_ROOT = "results/orn/rlb_1_22_c_lp/train/input/"
TEST_DESCRIPT_ROOT = "results/orn/rlb_1_22_c_lp/test/rlb_1_22_c_lp_descript.pickle"
TEST_IMG_ROOT = "results/orn/rlb_1_22_c_lp/test/input/"

class MemeDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []
        for img_path, captions in tqdm(data.items()):
            # caption_split = captions.split()
            # for i in range(len(caption_split)):
            #     self.img_paths.append(img_path)
            #     self.captions.append(caption_split[i])
            self.img_paths.append(img_path)
            # captions = captions.replace("circle", "zero")
            # captions = captions.replace("square", "four")
            # captions = captions.replace("triangle", "three")
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
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
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
                self.used_label_indices_count[class_] += self.n_samples
                # print('lenth of limit : ', len(self.label_to_indices[class_]))
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples  #self.n_samples == 1 fixed 

    def __len__(self):
        return self.n_dataset // self.batch_size

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

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

for epoch in range(EPOCH):
    print(f"running epoch {epoch}, best test loss {best_te_loss} after epoch {best_ep}")
    step = 0
    tr_loss = 0
    model.train()
    pbar = tqdm(train_dataloader, leave=False)
    for batch in pbar:
        step += 1
        optimizer.zero_grad()

        images = torch.zeros((BATCH_SIZE, 3, 224, 224))
        image_paths, texts, _ = batch
        for i, img_path in enumerate(image_paths):
            image = Image.open(img_path)
            image = image.resize((64,64))
            image = preprocess(image)
            images[i] = image

        images = images.to(device)
        texts = clip.tokenize(texts).to(device)
        # print(images.shape, texts.shape)
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(BATCH_SIZE).to(device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        tr_loss += total_loss.item()
        if device == "cpu":
            optimizer.step()
            scheduler.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            scheduler.step()
            clip.model.convert_weights(model)
        pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
    tr_loss /= step
    
    step = 0
    te_loss = 0
    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        for batch in test_pbar:
            step += 1

            images = torch.zeros((BATCH_SIZE, 3, 224, 224))
            image_paths, texts, _ = batch
            for i, img_path in enumerate(image_paths):
                image = Image.open(img_path)
                image = image.resize((64,64))
                image = preprocess(image)
                images[i] = image
                # images, texts, _ = batch

            images = images.to(device)
            texts = clip.tokenize(texts).to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(BATCH_SIZE).to(device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            te_loss += total_loss.item()
            test_pbar.set_description(f"test batchCE: {total_loss.item()}", refresh=True)
        te_loss /= step
        
    if te_loss < best_te_loss:
        best_te_loss = te_loss
        best_ep = epoch
        torch.save(model.state_dict(), "best_model_shape.pt")
    
    print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")
torch.save(model.state_dict(), "last_model.pt")



# for batch in train_dataloader:
#     imgs, txts, labels = batch
#     print(imgs[0])
#     print(len(txts))
#     print(labels)
#     print(labels.shape)
#     print(torch.unique(labels).shape)
#     break



# print(len(des_data[0]))



# image = preprocess(Image.open(TRAIN_IMG_ROOT)).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# print(image.shape)
# print(text.shape)

