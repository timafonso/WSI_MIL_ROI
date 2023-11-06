import numpy as np
import pandas as pd
# ================================== Torch Imports ==================================

import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn


# =================================== Metrics =======================================

from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryF1Score
from torcheval.metrics import BinaryAUROC, R2Score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, RocCurveDisplay

import matplotlib.pyplot as plt

# ============================= Models and Datasets =================================

from models.models import Attention, GatedAttention, AdditiveAttention, ModAdditiveAttention
from wsi_datasets import tumor_pad_collate_fn, gene_pad_collate_fn, GeneExpressionDataset, TumorEmbeddingDataset

# ============================ TensorBoard and Logging =============================
from torch.utils.tensorboard import SummaryWriter

#================================= Finetuning ======================================
import argparse

#==================================== PARAMETERS ====================================================

CUDA = True
SEED = 5

np.random.seed(SEED)

# --------------------------------- Metrics Indexes -----------------------------------------
LOSS_INDEX = 0
ACCURACY_INDEX = 1
AUC_INDEX = 2
RECALL_INDEX = 3
F1_INDEX = 4
PERCENTAGE_ACCURACY_INDEX = 5
# ------------------------------------- Labels ----------------------------------------------
LABELS = ["tumor", "tp53"]
# -------------------------------- Train/Test Split -----------------------------------------
TRAIN_PERC = 0.8
KFOLD_SPLITS = 5
# -------------------------------------------------------------------------------------------
CUDA_DEVICE = "cuda:0"
# ================================ Initializations ===============================================

parser = argparse.ArgumentParser(description='Finetuning Models.')
parser.add_argument("--hyperparameters_file", metavar="f", type=str)
parser.add_argument("--trial", metavar="t", type=int)
parser.add_argument("--lrsc", metavar="lrsc", type=int)
parser.add_argument("--tensorboard", metavar="tensorboard", type=str)
# parser.add_argument("--dataset", metavar="d", type=str)
# parser.add_argument("--output", metavar="out", type=str)

args = parser.parse_args()

df = pd.read_csv(args.hyperparameters_file)

model_str = args.hyperparameters_file.split("/")[2].split("_")[1]
dataset_str = args.hyperparameters_file.split("/")[1]
lrsc = int(args.lrsc)

task = "gene_expression" if dataset_str.split("_")[1] == "TP53" else "tumor_detection"

MODEL_WEIGHTS_FILE = "model_weights/{}/{}/baselines/{}_{}_{}.pt".format(task, dataset_str, model_str, args.trial, lrsc)
print(MODEL_WEIGHTS_FILE)

# dataset_path = args.dataset
# output_file = args.output 
# with open(args.hyperparameters, "r") as f:
#     hyperparameters = yaml.safe_load(f)

# CUDA initializations
torch.cuda.init()
torch.cuda.memory_summary(device=None, abbreviated=False)

torch.manual_seed(SEED)
cuda = CUDA and torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed_all(SEED)
    #torch.cuda.manual_seed(SEED)
    print('\nGPU is ON!')


device = torch.device(CUDA_DEVICE)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model to train
if model_str == "attention":
    model_type = Attention
elif model_str == "GatedAttention":
    model_type = GatedAttention
elif model_str == "additive":
    model_type = AdditiveAttention
else:      
    model_type = ModAdditiveAttention


epochs = df.loc[args.trial, "params_epochs"]
lr = df.loc[args.trial, "params_lr"]
weight_decay = df.loc[args.trial, "params_weight_decay"]
init_weights = df.loc[args.trial, "params_init_weights"]
log_batch_size = df.loc[args.trial, "params_log batch size"]


if args.tensorboard == "yes":
    USE_TENSORBOARD = True
else:
    USE_TENSORBOARD = False

dataset_dir = dataset_str.split("_")[0] + "_" + dataset_str.split("_")[2]
model_dir = "baselines/{}".format(model_str) if model_str != "mod" else "models/{}".format(model_str)
trial_dir = "trial_{}".format(str(args.trial))

TENSORBOARD_DIRECTORY = "runs/cross-val/{}/{}/{}/{}".format(task, dataset_dir, model_dir, trial_dir)

print(TENSORBOARD_DIRECTORY)

if USE_TENSORBOARD:
    writer = SummaryWriter(TENSORBOARD_DIRECTORY)

# Cross Validation
splits = KFold(n_splits = KFOLD_SPLITS, shuffle=True, random_state=42)
# ========================================== Metrics ======================================================

# -------------------------------- Metrics Intialization --------------------------

auc_metric = BinaryAUROC().to(device)    
accuracy_metric = BinaryAccuracy(threshold=0.5).to(device)
recall_metric = BinaryRecall(threshold=0.5).to(device)
F1_metric = BinaryF1Score(threshold=0.5).to(device)
R2Score_metric = R2Score()

# ----------------------------- Metrics Calculation Functions -----------------------

def calculate_r2_score(probs, true_labels):
    R2Score_metric.update(probs, true_labels)
    r2score = R2Score_metric.compute()
    R2Score_metric.reset()
    return r2score

def calculate_auc(probs, true_labels):
    auc_metric.update(probs, true_labels)
    auc = auc_metric.compute()
    auc_metric.reset()
    return auc

def calculate_accuracy(probs, true_labels):
    return accuracy_metric(probs, true_labels)

def calculate_recall(probs, true_labels):
    return recall_metric(probs, true_labels)

def calculate_F1(probs, true_labels):
    return F1_metric(probs, true_labels)


def trace_roc_curve(probs, true_labels):
    print(type(probs), type(true_labels))
    print(true_labels)
    true_labels = np.array([tensor.item() for tensor in true_labels])
    
    probs = np.array(probs)

    display = RocCurveDisplay.from_predictions(np.array(true_labels), probs)
    fig, ax = plt.subplots()
    display.plot(ax=ax)
    plt.savefig('roc_curve.png')


def calculate_metrics(probs, true_labels):
    probs = torch.FloatTensor(probs).to(device)
    true_labels = torch.FloatTensor(true_labels).to(device)
    
    accuracy = calculate_accuracy(probs, true_labels)
    auc = calculate_auc(probs, true_labels)
    f1 = calculate_F1(probs, true_labels)
    recall = calculate_recall(probs, true_labels)
    
    return accuracy, auc, f1, recall

# --------------------------- Metrics Presentation Functions ---------------------

def get_classification_report(probs, true_labels):
    report = classification_report(true_labels, probs, target_names=["Negative", "Positive"])
    print(report)

def metrics_to_tensorboard(fold_metrics):
    if USE_TENSORBOARD:
        for fold in range(fold_metrics.shape[0]):
            for epoch in range(fold_metrics.shape[1]):
                writer.add_scalar('{} Loss - Fold {}'.format("training", fold), fold_metrics[fold, epoch, 0, LOSS_INDEX], epoch)
                writer.add_scalar('{} Accuracy - Fold {}'.format("training", fold), fold_metrics[fold, epoch, 0, ACCURACY_INDEX], epoch)
                writer.add_scalar('{} AUC - Fold {}'.format("training", fold), fold_metrics[fold, epoch, 0, AUC_INDEX], epoch)
                writer.add_scalar('{} Recall - Fold {}'.format("training", fold), fold_metrics[fold, epoch, 0, RECALL_INDEX], epoch)
                writer.add_scalar('{} F1 - Fold {}'.format("training", fold), fold_metrics[fold, epoch, 0, F1_INDEX], epoch)
                
                writer.add_scalar('{} Loss - Fold {}'.format("validation", fold), fold_metrics[fold, epoch, 1, LOSS_INDEX], epoch)
                writer.add_scalar('{} Accuracy - Fold {}'.format("validation", fold), fold_metrics[fold, epoch, 1, ACCURACY_INDEX], epoch)
                writer.add_scalar('{} AUC - Fold {}'.format("validation", fold), fold_metrics[fold, epoch, 1, AUC_INDEX], epoch)
                writer.add_scalar('{} Recall - Fold {}'.format("validation", fold), fold_metrics[fold, epoch, 1, RECALL_INDEX], epoch)
                writer.add_scalar('{} F1 - Fold {}'.format("validation", fold), fold_metrics[fold, epoch, 1, F1_INDEX], epoch)
                writer.add_scalar('{} Percentages - Fold {}'.format("validation", fold), fold_metrics[fold, epoch, 1, PERCENTAGE_ACCURACY_INDEX], epoch) 


        for epoch in range(fold_metrics.shape[1]):
            writer.add_scalar('{} Loss - Fold {}'.format("training", "avg"), np.mean(fold_metrics[:, epoch, 0, LOSS_INDEX], axis=0), epoch)
            writer.add_scalar('{} Accuracy - Fold {}'.format("training", "avg"), np.mean(fold_metrics[:, epoch, 0, ACCURACY_INDEX], axis=0), epoch)
            writer.add_scalar('{} AUC - Fold {}'.format("training", "avg"), np.mean(fold_metrics[:, epoch, 0, AUC_INDEX], axis=0), epoch)
            writer.add_scalar('{} Recall - Fold {}'.format("training", "avg"), np.mean(fold_metrics[fold, epoch, 0, RECALL_INDEX]), epoch)
            writer.add_scalar('{} F1 - Fold {}'.format("training", "avg"), np.mean(fold_metrics[fold, epoch, 0, F1_INDEX]), epoch)   
            
            writer.add_scalar('{} Loss - Fold {}'.format("validation", "avg"), np.mean(fold_metrics[:, epoch, 1, LOSS_INDEX], axis=0), epoch)
            writer.add_scalar('{} Accuracy - Fold {}'.format("validation", "avg"), np.mean(fold_metrics[:, epoch, 1, ACCURACY_INDEX], axis=0), epoch)
            writer.add_scalar('{} AUC - Fold {}'.format("validation", "avg"), np.mean(fold_metrics[:, epoch, 1, AUC_INDEX], axis=0), epoch)
            writer.add_scalar('{} Recall - Fold {}'.format("validation", "avg"), np.mean(fold_metrics[:, epoch, 1, RECALL_INDEX]), epoch)
            writer.add_scalar('{} F1 - Fold {}'.format("validation", "avg"), np.mean(fold_metrics[:, epoch, 1, F1_INDEX]), epoch) 
            writer.add_scalar('{} Percentages - Fold {}'.format("validation", "avg"), np.mean(fold_metrics[:, epoch, 1, PERCENTAGE_ACCURACY_INDEX]), epoch)

    return np.mean(fold_metrics[:, -1, 1, LOSS_INDEX]), np.mean(fold_metrics[:, -1, 1, ACCURACY_INDEX]), np.mean(fold_metrics[:, -1, 1, AUC_INDEX])   


def external_validation_to_tensorboard(loss, auc, accuracy, f1, recall, dataloader_len):
    writer.add_text('loss', "{:.4f}".format(loss/dataloader_len), 0)
    writer.add_text('auc', "{:.4f}".format(auc), 0)
    writer.add_text('accuracy',"{:.4f}".format(accuracy), 0)
    writer.add_text('f1',"{:.4f}".format(f1), 0)
    writer.add_text('recall',"{:.4f}".format(recall), 0)

def remove_padding(data):
    mask = np.all(np.array(data) != -np.Inf, axis=1)
    return data[mask]


# =========================================== Dataset ======================================================
# Initialiation
if task == "tumor_detection":
    dataset = TumorEmbeddingDataset("datasets/{}/{}.hdf5".format(task, dataset_str))
    collate_fn = tumor_pad_collate_fn
else:
    dataset = GeneExpressionDataset("datasets/{}/{}.hdf5".format(task, dataset_str), "tp53")
    collate_fn = gene_pad_collate_fn


# Train/Test Split
train_size = int(TRAIN_PERC * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# -------------------------------------- Auxiliary Functions ---------------------------------------------

def remove_padding(data):
    mask = np.all(np.array(data) != -np.Inf, axis=1)
    return data[mask]

# ===================================== Train Function ========================================

# Train for a Single Batch
def train_batch(model, data, labels):
    losses, Y_probs, true_labels = [], [], []
    num_nan = 0
    train_error = 0
    train_correct = 0
    
    for data_element, label in zip(data, labels):
        if torch.isnan(data_element).any():
            num_nan += 1
            continue
        data_element = remove_padding(data_element)
        if cuda:
            data_element, label = data_element.to(device), label.to(device)
        
        loss, attention, scores = model.calculate_objective(data_element, label)
        error, Y_hat, Y_prob    = model.calculate_classification_error(data_element, label)

        if torch.isnan(Y_prob).any():
            num_nan += 1
            continue
        losses.append(loss)
        train_error += error
        train_correct += (Y_hat == label).sum().item()
        Y_probs.append(Y_prob.item())
        true_labels.append(label)

    return losses, train_error, train_correct, Y_probs, true_labels, num_nan


# Full epoch training
def train_epoch_embeddings(model, dataloader, optimizer, epoch, fold=""):
    model.train()
    train_loss, train_error, train_correct = 0., 0., 0.
    Y_probs = []
    true_labels = []
    num_nan = 0
    
    for batch_idx, (data, _, labels, slide_ids, _, _) in enumerate(dataloader):
        batch_losses, batch_error, batch_correct, batch_y_probs, batch_labels, batch_nan = train_batch(model, data, labels)
        final_loss = torch.stack(batch_losses)
        train_loss += final_loss.sum()
        train_error += batch_error
        train_correct += batch_correct
        Y_probs += batch_y_probs
        batch_nan += batch_nan
        true_labels += batch_labels
        final_loss.mean().backward()
        optimizer.step()

    return train_loss, train_correct, Y_probs, true_labels, num_nan




# ====================================== Validation Functions =====================================

def valid_batch(model, data, slide_ids, case_ids, labels, augmentation_flags, score_list):
    valid_loss, valid_error, valid_correct = 0, 0, 0
    Y_probs, true_labels = [], []
    num_nan = 0

    for data_element, label, slide_id, is_aug in zip(data, labels, slide_ids, augmentation_flags):
        if is_aug:
            break
        data_element = remove_padding(data_element)

        if cuda:
            data_element, label = data_element.to(device), label.to(device)
      
        loss, attention, scores = model.calculate_objective(data_element, label)
        error, Y_hat, Y_prob    = model.calculate_classification_error(data_element, label)
        
        score_list[slide_id] = scores 
        if torch.any(torch.isnan(Y_prob)):
            num_nan += 1
            continue
        valid_loss += loss.item()
        valid_error += error
        valid_correct += (Y_hat == label).sum().item()
        Y_probs.append(Y_prob.item())
        true_labels.append(label)

    return valid_loss, valid_correct, Y_probs, true_labels,  num_nan, score_list


def valid_epoch_embeddings(model, dataloader, epoch, fold=""):
    model.eval()
    valid_loss, valid_error, valid_correct = 0., 0., 0
    true_labels = []
    Y_probs = []
    num_nan = 0
    score_list = {}

    for batch_idx, (data, coords, labels, slide_ids, case_ids, aug_flags) in enumerate(dataloader):
        batch_loss, batch_correct, batch_y_probs, batch_labels, batch_nan, score_list = valid_batch(model, data, slide_ids, case_ids, labels, aug_flags, score_list)
        valid_loss += batch_loss
        valid_correct += batch_correct
        Y_probs += batch_y_probs
        num_nan += batch_nan
        true_labels += batch_labels
        

    
    if epoch == "validation":
        accuracy, auc, f1, recall = calculate_metrics(Y_probs, true_labels)
        trace_roc_curve(Y_probs, true_labels)
        if USE_TENSORBOARD:
            external_validation_to_tensorboard(valid_loss, auc, accuracy, f1, recall, len(dataloader))
        return valid_loss, valid_correct, Y_probs, true_labels, score_list, auc
        
    return valid_loss, valid_correct, Y_probs, true_labels, score_list


# ============================ Cross Validation ===============================


# Weights Initialization Function
def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0)

def model_initialization(model_type, init_weights):   
    model = model_type()
    model = model.to(device)
    if init_weights == "xavier":
        model.apply(weights_init_xavier)
    else:
        model.apply(weights_init_kaiming)    
    return model

def cross_validation(model_type, dataset, optimizer_type, lr, weight_decay, batch_size=1, num_epochs=15, init_weights="xavier"):
    print(lr, weight_decay)
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    fold_metrics = np.empty((KFOLD_SPLITS, num_epochs, 2, 6))
    models = []
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        torch.manual_seed(SEED)
        model = model_initialization(model_type, init_weights)
        optimizer = optimizer_type(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        print('Fold {}'.format(fold + 1))
        train_sampler = data_utils.SubsetRandomSampler(train_idx)
        test_sampler = data_utils.SubsetRandomSampler(val_idx)
        #print(batch_size, type(batch_size))
        train_loader = data_utils.DataLoader(dataset, batch_size, sampler=train_sampler, pin_memory=True, collate_fn=collate_fn, num_workers=0)
        test_loader = data_utils.DataLoader(dataset, batch_size, sampler=test_sampler, pin_memory=True, collate_fn=collate_fn, num_workers=0)

        train_auc_epochs, train_accuracy_epochs, train_loss_epochs = [], [], []
        test_auc_epochs, test_accuracy_epochs, test_loss = [], [], []
        
        
        if lrsc != -1:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lrsc)
        
        for epoch in range(0, num_epochs):
            train_loss, train_correct, train_probs, train_true_labels, num_nan = train_epoch_embeddings(model,train_loader, optimizer, epoch, fold)
            test_loss, test_correct, test_probs, test_true_labels, _  = valid_epoch_embeddings(model,test_loader, epoch, fold)
            
            train_accuracy, train_auc, train_f1, train_recall = calculate_metrics(train_probs, train_true_labels)
            test_accuracy, test_auc, test_f1, test_recall = calculate_metrics(test_probs, test_true_labels)  


            num_bags_per_sample = train_dataset[0][0].shape[0] 

            fold_metrics[fold, epoch, 0, LOSS_INDEX] = train_loss / (len(train_sampler) * num_bags_per_sample - num_nan)
            fold_metrics[fold, epoch, 0, ACCURACY_INDEX] = train_accuracy
            fold_metrics[fold, epoch, 0, AUC_INDEX] = train_auc
            fold_metrics[fold, epoch, 0, F1_INDEX] = train_f1
            fold_metrics[fold, epoch, 0, RECALL_INDEX] = train_recall
            
            fold_metrics[fold, epoch, 1, LOSS_INDEX] = test_loss / len(test_loader)
            fold_metrics[fold, epoch, 1, ACCURACY_INDEX] = test_accuracy
            fold_metrics[fold, epoch, 1, AUC_INDEX] = test_auc
            fold_metrics[fold, epoch, 1, F1_INDEX] = test_f1
            fold_metrics[fold, epoch, 1, RECALL_INDEX] = test_recall

            if lrsc != -1:
                lr_scheduler.step()
        models.append((fold_metrics[fold, -1, 1, AUC_INDEX], model))

    chosen_model = None
    max_auc = 0

    for model in models:
        if model[0] > max_auc:
            max_auc = model[0]
            chosen_model = model[1]

    loss, accuracy, auc = metrics_to_tensorboard(fold_metrics)

    return chosen_model, loss, accuracy, auc    

optimizer_type = optim.Adam
seeds = [2,3,5,7,11]

losses, aucs = [], []
for seed in seeds:
    train_size = int(TRAIN_PERC * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    model, loss, accuracy, auc = cross_validation(model_type, train_dataset, optimizer_type, lr, weight_decay, int(2**log_batch_size), int(epochs), init_weights)

    dataloader = data_utils.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True, collate_fn=gene_pad_collate_fn)
    valid_loss, valid_correct, probs, true_labels, scores, auc = valid_epoch_embeddings(model, dataloader, "validation")
    aucs.append(auc.cpu().detach().numpy())
    losses.append(valid_loss)
    print(aucs)