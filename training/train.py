
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
import warnings
import time
import pickle
import itertools
import seaborn as sns
import pandas as pd
warnings.filterwarnings("ignore")


class Training:

    def __init__(self, name, train_transform, val_transform, batchsize, lr, epochs, model, criterions, optimizers, scheduler=None, custom_datasets=None, version="pt"):
        self.classes = ['Angry', 'Disgust', 'Fear',
                        'Happy', 'Sad', 'Surprise', 'Neutral']
        self.name = name
        self.folder = [name.split(
            "_")[1]] if custom_datasets is None else name.split("_")[1:]
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batchsize = batchsize
        self.lr = lr
        self.epochs = epochs
        self.custom_datasets = custom_datasets
        self.init_dataloaders()
        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else None
        self.model = model
        self.model.to(self.device)
        self.criterions = criterions
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.history = {}
        self.version = version
        print(f"{self.name} is initialized and ready to train.")

    def init_dataloaders(self):
        self.fer2013_dataset = ImageFolder(
            'FER2013', self.val_transform)
        self.jaffe_dataset = ImageFolder('JAFFE', self.val_transform)
        self.fer2013_loader = DataLoader(
            self.fer2013_dataset, self.batchsize, num_workers=6, pin_memory=True)
        self.jaffe_loader = DataLoader(
            self.jaffe_dataset, self.batchsize, num_workers=6, pin_memory=True)
        if self.custom_datasets is None:
            self.train_dataset = ImageFolder(
                self.folder[0]+'/train', self.train_transform)
            self.val_dataset = ImageFolder(
                self.folder[0]+'/validation', self.val_transform)
        else:
            self.train_dataset = self.custom_datasets[0]
            self.val_dataset = self.custom_datasets[1]
        self.train_loader = DataLoader(
            self.train_dataset, self.batchsize, shuffle=True, num_workers=6, pin_memory=True)
        self.val_loader = DataLoader(
            self.val_dataset, self.batchsize, num_workers=6, pin_memory=True)

    def train(self):
        if self.device == None:
            print("CPU Training not supported. Please use GPU.")
        else:
            self.model.to(self.device)
            start = time.time()
            print(
                f"===================================Start {self.name} training===================================")
            for e in range(self.epochs):
                train_loss = 0
                validation_loss = 0
                train_correct = 0
                val_correct = 0
                self.model.train()
                learning_rate = []
                for data, labels in self.train_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    for o in self.optimizers:
                        o.zero_grad()
                    if self.name == "DAN_RAF-DB":
                        outputs, feat, heads = self.model(data)
                        loss = self.criterions[0](
                            outputs, labels) + self.criterions[1](feat, labels) + self.criterions[2](heads)
                    elif self.name.startswith("DACL"):
                        feat, outputs, A = self.model(data)
                        l_softmax = self.criterions[0](outputs, labels)
                        l_center = self.criterions[1](
                            feat.cpu(), A.cpu(), labels.cpu())
                        loss = l_softmax + 0.01 * l_center.to(self.device)
                    else:
                        outputs = self.model(data)
                        loss = self.criterions[0](outputs, labels)
                    loss.backward()
                    for o in self.optimizers:
                        o.step()
                    for param_group in self.optimizers[0].param_groups:
                        learning_rate.append(param_group['lr'])
                    train_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    train_correct += torch.sum(preds == labels.data)

                self.model.eval()
                for data, labels in self.val_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    val_outputs = self.model(data)
                    if self.name == "DAN_RAF-DB":
                        val_outputs, feat, heads = self.model(data)
                        val_loss = self.criterions[0](
                            val_outputs, labels) + self.criterions[1](feat, labels) + self.criterions[2](heads)
                    elif self.name.startswith("DACL"):
                        feat, val_outputs, A = self.model(data)
                        l_softmax = self.criterions[0](val_outputs, labels)
                        l_center = self.criterions[1](
                            feat.cpu(), A.cpu(), labels.cpu())
                        val_loss = l_softmax + 0.01 * l_center.to(self.device)
                    else:
                        val_outputs = self.model(data)
                        val_loss = self.criterions[0](val_outputs, labels)
                    validation_loss += val_loss.item()
                    _, val_preds = torch.max(val_outputs, 1)
                    val_correct += torch.sum(val_preds == labels.data)

                if self.scheduler is not None:
                    self.scheduler.step()
                train_loss = train_loss/len(self.train_dataset)
                train_acc = train_correct.double() / len(self.train_dataset)
                validation_loss = validation_loss / len(self.val_dataset)
                val_acc = val_correct.double() / len(self.val_dataset)
                self.history[e+1] = [train_loss, train_acc,
                                     validation_loss, val_acc, learning_rate]
                print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                      .format(e+1, train_loss, validation_loss, train_acc * 100, val_acc*100))
            torch.save(self.model.state_dict(),
                       'Checkpoints/{}.pt'.format(self.name))
            end = time.time()
            with open('Checkpoints/History_{}.pickle'.format(self.name), 'wb') as handle:
                pickle.dump(self.history, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            print("==================Training {} finished in {} with {:.3f}%==================".format(
                self.name, end-start, val_acc*100))

    def plot_accuracies(self):
        if len(self.history) == 0:
            try:
                with open(f"Checkpoints/History_{self.name}.pickle", "rb") as h:
                    self.history = pickle.load(h)
            except:
                print("You need to train the model before you can plot graphs for it.")
        if len(self.history) > 0:
            train_acc = [v[1].cpu() for v in self.history.values()]
            val_acc = [v[3].cpu() for v in self.history.values()]
            plt.figure(figsize=(16, 8))
            plt.plot(train_acc, '-b', label="train_acc")
            plt.plot(val_acc, '-r', label="val_acc")
            plt.legend()
            plt.xticks(np.arange(0, len(train_acc), 20))
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid()
            plt.title('Accuracy vs. Epochs')

    def plot_losses(self):
        if len(self.history) == 0:
            try:
                with open(f"Checkpoints/History_{self.name}.pickle", "rb") as h:
                    self.history = pickle.load(h)
            except:
                print("You need to train the model before you can plot graphs for it.")
        if len(self.history) > 0:
            train_losses = [v[0] for v in self.history.values()]
            val_losses = [v[2] for v in self.history.values()]
            plt.figure(figsize=(16, 8))
            plt.plot(train_losses, '-b', label="train_loss")
            plt.plot(val_losses, '-r', label="val_loss")
            plt.legend()
            plt.xticks(np.arange(0, len(train_losses), 20))
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.grid()
            plt.title('Loss vs. Epochs')

    def plot_lrs(self):
        if len(self.history) == 0:
            try:
                with open(f"Checkpoints/History_{self.name}.pickle", "rb") as h:
                    self.history = pickle.load(h)
            except:
                print("You need to train the model before you can plot graphs for it.")
        if len(self.history) > 0:
            lrs = np.concatenate([v[4] for v in self.history.values()])
            plt.figure(figsize=(16, 8))
            plt.plot(lrs)
            plt.xlabel('Batch no.')
            plt.ylabel('Learning rate')
            plt.title('Learning Rate vs. Batch no.')

    def show_emotion_distribution(self):
        count_emotion = {}
        sum_data = 0
        for emotion in self.classes:
            x = 0
            for f in self.folder:
                x = x+len(os.listdir(f + "/train/" + emotion.lower()))
            count_emotion[emotion.lower()] = x
            sum_data = sum_data+x
        plt.figure(figsize=(6, 6))
        plt.bar(count_emotion.keys(), count_emotion.values(), color=[
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
        plt.title('Emotion Distribution')
        plt.ylabel("Count")
        print(f"Total number of train images: {sum_data}")

    def show_batch(self, dl, title):
        for images, _ in dl:
            _, ax = plt.subplots(figsize=(6, 6))
            plt.title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images[:36], nrow=6).permute(1, 2, 0))
            break

    def show_transform_impact(self):
        if self.custom_datasets is None:
            no_transform = tt.Compose([tt.Resize((100, 100)), tt.ToTensor()])
            imfo = ImageFolder(self.folder[0]+'/train', no_transform)
            without_transform = DataLoader(
                imfo, self.batchsize, shuffle=True, num_workers=6, pin_memory=True)
            self.show_batch(without_transform, "Before transformation")
            self.show_batch(self.train_loader, "After transformation")
        else:
            print("Cant show transformation impact of custom datasets.")

    def test_data(self, testname, dataloader):
        try:
            correct_pred = {classname: 0 for classname in self.classes}
            total_pred = {classname: 0 for classname in self.classes}
            confusion_matrix = torch.zeros(7, 7)
            if self.version == "pth":
                self.model.load_state_dict(torch.load(
                    "Checkpoints/{}.pth".format(self.name))["model_state_dict"])
            else:
                self.model.load_state_dict(torch.load(
                    "Checkpoints/{}.pt".format(self.name)))
            self.model.eval()
            with torch.no_grad():
                for data, labels in dataloader:
                    data, labels = data.cuda(), labels.cuda()
                    if self.name == "DAN_RAF-DB":
                        outputs = self.model(data)[0]
                    elif self.name.startswith("DACL"):
                        outputs = self.model(data)[1]
                    else:
                        outputs = self.model(data)
                    _, predictions = torch.max(outputs, 1)
                    for label, prediction in zip(labels, predictions):
                        if label == prediction:
                            correct_pred[self.classes[label]] += 1
                        confusion_matrix[label.view(-1).long(),
                                         prediction.view(-1).long()] += 1
                        total_pred[self.classes[label]] += 1

            print(
                f'Accuracy on the {sum(total_pred.values())} images from {testname}: {100 * sum(correct_pred.values()) // sum(total_pred.values())} %')

            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(
                    f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
            plt.figure(figsize=(15, 10))

            df_cm = pd.DataFrame(
                confusion_matrix, index=self.classes, columns=self.classes).astype(int)
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
            heatmap.yaxis.set_ticklabels(
                heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
            heatmap.xaxis.set_ticklabels(
                heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        except Exception:
            print(
                "Either you haven't trained the model yet or an error occurred during testing.")

    def test_jaffe(self):
        if self.device == None:
            print("Testing with CPU is not supported. Please use GPU.")
        else:
            self.test_data("JAFFE", self.jaffe_loader)

    def test_fer2013(self):
        if self.device == None:
            print("Testing with CPU is not supported. Please use GPU.")
        else:
            self.test_data("FER2013", self.fer2013_loader)
