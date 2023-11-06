import torch
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.init()
device = torch.device("cuda:0")
import numpy as np

#======================================== Attention Model ===========================================================

class Attention(nn.Module):
    def __init__(self, self_attention=False):
        super(Attention, self).__init__()
        self.L = 1024
        self.D = 128
        self.K = 1
        self.self_attention = self_attention

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        if self.self_attention:
            self.self_att = SelfAttention(self.L)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.self_attention:
            x, _, _, _ = self.self_att(x)
        A = self.attention(x)  # NxK
        A = A.to(device)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, x)  # KxL
        M = M.to(device)
        Y_prob = self.classifier(M)
        Y_prob = Y_prob.to(device)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y = Y.to(device)
        Y_prob, Y_hat, _ = self.forward(X)
        Y_prob, Y_hat = Y_prob.to(device), Y_hat.to(device)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y):
        Y = Y.to(device)
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob, A = Y_prob.to(device), A.to(device)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        neg_log_likelihood = neg_log_likelihood.to(device)

        return neg_log_likelihood, A, 0



class GatedAttention(nn.Module):
    def __init__(self, self_attention=False):
        super(GatedAttention, self).__init__()
        self.L = 1024
        self.D = 128
        self.K = 1
        self.self_attention = self_attention

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        if self.self_attention:
            self.self_att = SelfAttention(self.L)

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, H):
        if self.self_attention:
            H, _, _, _ = self.self_att(H)
        A_V = self.attention_V(H) 
        A_U = self.attention_U(H)  
        A = self.attention_weights(A_V * A_U) 
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        return neg_log_likelihood, A, 0


#===================================== Self-Attention Layer ===============================================

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter((torch.zeros(1)).cuda())
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter((torch.ones(1)).cuda())

    def forward(self, x):
        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x
        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att


# ========================================== Additive Model =========================================================

class AdditiveAttention(nn.Module):
    def __init__(self):
        super(AdditiveAttention, self).__init__()
        self.L = 1024
        self.D = 256
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.BatchNorm1d(self.D, track_running_stats=True),
            nn.LeakyReLU(0.2),
            nn.Linear(self.D, self.D),
            nn.LeakyReLU(0.2),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.D),
            nn.ReLU(),
            nn.Linear(self.D, 2)
        )

    def forward(self, x):
        A = self.attention(x)
        A = A.to(device)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mul(A.T, x)
        scores = self.classifier(M)
        bag_score = torch.sum(scores, dim=0)
        probs = F.softmax(bag_score, dim=0)
        Y_prob = probs[1]
        Y_hat = torch.argmax(probs)
        return Y_prob, Y_hat, A, scores

    def calculate_objective(self, X, Y, tissue_perc=None):
        if tissue_perc != None:
            tissue_perc = tissue_perc.to(device)
        Y = Y.to(device)
        Y = Y.float()
        Y_prob, Y_hat, A, scores = self.forward(X)
        Y_prob = Y_prob.to(device)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

        scaled_scores = F.sigmoid(scores[:,1])

        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob)) 
        neg_log_likelihood = neg_log_likelihood.to(device)

        return neg_log_likelihood.reshape(1), A, scaled_scores

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y = Y.to(device)
        Y_prob, Y_hat, _, _ = self.forward(X)
        Y_prob, Y_hat = Y_prob.to(device), Y_hat.to(device)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, Y_prob




class ModAdditiveAttention(nn.Module):
    def __init__(self):
        super(ModAdditiveAttention, self).__init__()
        self.L = 1024
        self.D = 256
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.D),
            nn.ReLU(),
            nn.Linear(self.D, 2)
        )

    def forward(self, x):
        A = self.attention(x)  # NxK
        A = A.to(device)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mul(A.T, x)
        scores = self.classifier(M)
        bag_score = torch.sum(scores, dim=0)
        probs = F.softmax(bag_score, dim=0)
        Y_prob = probs[1]
        if Y_prob >= 0.4:
            Y_hat = torch.tensor(1)
        else:
            Y_hat = torch.tensor(0)
        scaled_scores = F.sigmoid(scores[:,1])
        mean_scores = (scaled_scores >= 0.5).float().mean()
        return Y_prob, Y_hat, A, scaled_scores, scores[:,1]

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y = Y.to(device)
        Y_prob, Y_hat, _, _, _ = self.forward(X)
        Y_prob, Y_hat = Y_prob.to(device), Y_hat.to(device)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y, validation=False):
        Y = Y.to(device)
        Y = Y.float()
        Y_prob, Y_hat, A, scaled_scores, mean_scores = self.forward(X)
        Y_prob, A = Y_prob.to(device), A.to(device)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob)) 
        neg_log_likelihood = neg_log_likelihood.to(device)

        return neg_log_likelihood, A, scaled_scores


