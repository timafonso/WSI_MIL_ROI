import numpy as np

# ================================== Torch Imports ==================================

import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn

# =================================== Metrics =======================================

from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryF1Score
from torcheval.metrics import BinaryAUROC, R2Score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

# ============================= Models and Datasets =================================

from models.models import Attention, GatedAttention, AdditiveAttention, ModAdditiveAttention
from wsi_datasets import tumor_pad_collate_fn, gene_pad_collate_fn, GeneExpressionDataset, TumorEmbeddingDataset

# ============================ TensorBoard and Logging =============================
from torch.utils.tensorboard import SummaryWriter


#================================= Finetuning ======================================

import optuna
import yaml
import argparse

#==================================== PARAMETERS ====================================================

CUDA = True
SEED = 5

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

# ================================ Initializations ===============================================

parser = argparse.ArgumentParser(description='Finetuning Models.')
parser.add_argument("--hyperparameters", metavar="f", type=str)
parser.add_argument("--dataset", metavar="d", type=str)
parser.add_argument("--output", metavar="out", type=str)
parser.add_argument("--cuda", metavar="cuda", type=str)


args = parser.parse_args()

CUDA_DEVICE = args.cuda

dataset_path = args.dataset
output_file = args.output 
with open(args.hyperparameters, "r") as f:
    hyperparameters = yaml.safe_load(f)

# CUDA initializations
torch.cuda.init()
torch.cuda.memory_summary(device=None, abbreviated=False)
cuda = CUDA and torch.cuda.is_available()
device = torch.device(CUDA_DEVICE)
torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)
print('\nGPU is ON!')

np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Model to train
if hyperparameters["model"] == "Attention":
    model_type = Attention
elif hyperparameters["model"] == "GatedAttention":
    model_type = GatedAttention
elif hyperparameters["model"] == "AdditiveAttention":
    model_type = AdditiveAttention
else:      
    model_type = ModAdditiveAttention

log_batch_size = hyperparameters["log_batch_size"]
epoch = hyperparameters["epochs"]
lr = hyperparameters["learning_rate"]
weight_decay = hyperparameters["weight_decay"]
feature_selection = hyperparameters["feature_selection"]
task = hyperparameters["task"]

USE_TENSORBOARD = False
TENSORBOARD_DIRECTORY = "runs/cross-val/gene_expression/BRCA_10x/baselines/Attention/trial_133_test"
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

def get_metrics_matrix(fold_metrics):
    return np.mean(fold_metrics[:, -1, 1, LOSS_INDEX]), np.mean(fold_metrics[:, -1, 1, ACCURACY_INDEX]), np.mean(fold_metrics[:, -1, 1, AUC_INDEX])   

def remove_padding(data):
    mask = np.all(np.array(data) != -np.Inf, axis=1)
    return data[mask]


# =========================================== Dataset ======================================================

# Initialiation
if task == "label":
    dataset = TumorEmbeddingDataset(dataset_path)
    collate_fn = tumor_pad_collate_fn
else:
    dataset = GeneExpressionDataset(dataset_path, task)
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
        
        loss, _, _ = model.calculate_objective(data_element, label)
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
      
        loss, _, _ = model.calculate_objective(data_element, label)
        error, Y_hat, Y_prob    = model.calculate_classification_error(data_element, label)
    
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
    valid_loss, valid_correct = 0., 0
    true_labels = []
    Y_probs = []
    num_nan = 0
    score_list = {}

    for _, (data, _, labels, slide_ids, case_ids, aug_flags) in enumerate(dataloader):
        batch_loss, batch_correct, batch_y_probs, batch_labels, batch_nan, score_list = valid_batch(model, data, slide_ids, case_ids, labels, aug_flags, score_list)
        valid_loss += batch_loss
        valid_correct += batch_correct
        Y_probs += batch_y_probs
        num_nan += batch_nan
        true_labels += batch_labels
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
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        torch.manual_seed(SEED)
        model = model_initialization(model_type, init_weights)
        optimizer = optimizer_type(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        print('Fold {}'.format(fold + 1))
        train_sampler = data_utils.SubsetRandomSampler(train_idx)
        test_sampler = data_utils.SubsetRandomSampler(val_idx)
        train_loader = data_utils.DataLoader(dataset, batch_size, sampler=train_sampler, pin_memory=True, collate_fn=collate_fn)
        test_loader = data_utils.DataLoader(dataset, batch_size, sampler=test_sampler, pin_memory=True, collate_fn=collate_fn)
        
        for epoch in range(0, num_epochs):
            train_loss, _, train_probs, train_true_labels, num_nan = train_epoch_embeddings(model,train_loader, optimizer, epoch, fold)
            test_loss, _, test_probs, test_true_labels, _  = valid_epoch_embeddings(model,test_loader, epoch, fold)
            
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

            
    loss, accuracy, auc = get_metrics_matrix(fold_metrics)

    return model, loss, accuracy, auc    


def objective(trial):
    """
    Hyperparameter fine-tuning objective function.

    - Learning rate
    - Weight Decaying
    - Batch Size
    - Epochs
    - Feature Selection
    - Optimizer
    """
    learning_rate = trial.suggest_float("lr", lr[0], lr[1], log=True)
    weight_dec = trial.suggest_float("weight_decay", weight_decay[0], weight_decay[1], log=True)
    batch_size = trial.suggest_int("log batch size", log_batch_size[0], log_batch_size[1])
    epochs = trial.suggest_int("epochs", epoch[0], epoch[1], 5)
    init_weights = trial.suggest_categorical("init_weights", ["xavier", "kaiming"])

    optimizer_type = optim.Adam

    _, loss, accuracy, auc = cross_validation(model_type, train_dataset, optimizer_type, learning_rate, weight_dec, 2**batch_size, epochs, init_weights)

    return loss, auc

study = optuna.create_study(directions=["minimize", "maximize"])
study.optimize(objective, n_trials=100, n_jobs=1)

print("Best trials:")
trials = study.best_trials
print("  Params: ")
for trial in trials:
    print("trial", trial.number)
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

df = study.trials_dataframe()
df.to_csv(output_file)
print("hyperparameters file", args.hyperparameters)
print("dataset used", dataset_path)
print("results in", output_file)
