import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
from sklearn.cluster import KMeans

# Plot metrics
def plot_metrics(y_true, y_pred, title="Model Performance"):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_score(y_true, y_pred):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure()
    plt.plot(recall, precision, label=f"Precision-Recall curve (area = {auc(recall, precision):.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.show()

class ProteinPairsDataset(Dataset):
    def __init__(self, file_path, reduction_factor=0.5, use_knn_reduction=False, num_clusters=1000):
        data = pd.read_csv(file_path, sep='\t')
        self.geometric_indices = list(range(1, 18))
        self.zernike_indices = list(range(18, 393))
        
        if use_knn_reduction and reduction_factor < 1.0:
            sample_size = int(len(data) * reduction_factor)
            features = data.iloc[:, self.geometric_indices + self.zernike_indices].values
            kmeans = KMeans(n_clusters=min(num_clusters, sample_size), random_state=42)
            data['cluster'] = kmeans.fit_predict(features)
            sampled_data = data.groupby('cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)
            self.data = sampled_data.values
        else:
            self.data = data.values

        self.labels = (self.data[:, 0][:, None] == self.data[:, 0]).astype(int)

    def __len__(self):
        return len(self.data) ** 2

    def __getitem__(self, idx):
        i, j = divmod(idx, len(self.data))
        g1, g2 = self.data[i, self.geometric_indices], self.data[j, self.geometric_indices]
        z1, z2 = self.data[i, self.zernike_indices], self.data[j, self.zernike_indices]

        geom_dist = np.linalg.norm(g1 - g2)
        zernike_dist = np.linalg.norm(z1 - z2)

        label = self.labels[i, j]
        return torch.tensor([geom_dist, zernike_dist], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def get_dataloader(file_path, batch_size=64, weighted_sampling=True, reduction_factor=0.5, use_knn_reduction=False, num_clusters=1000):
    dataset = ProteinPairsDataset(file_path, reduction_factor=reduction_factor, use_knn_reduction=use_knn_reduction, num_clusters=num_clusters)
    if weighted_sampling:
        class_counts = np.bincount(dataset.labels.flatten())
        weights = 1.0 / class_counts[dataset.labels.flatten()]
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

class ProteinNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(ProteinNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  
        return x.squeeze()

batch_size = 128
hidden_dim = 64
learning_rate = 0.001
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_dataloader("/kaggle/input/rcsb-data/cath_moments.tsv", batch_size=batch_size, reduction_factor=0.5, use_knn_reduction=True, num_clusters=1000)
eval_loader = get_dataloader("/kaggle/input/rcsb-data/ecod_moments.tsv", batch_size=batch_size, weighted_sampling=False)
model = ProteinNN(hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
writer = SummaryWriter("/kaggle/working/runs/clustering_and_reduction")

def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).detach().cpu().numpy() 
            y_pred.extend(outputs)
            y_true.extend(labels.cpu().numpy())

    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    best_mcc = matthews_corrcoef(y_true, (np.array(y_pred) > 0.5).astype(int))
    return roc_auc, pr_auc, best_mcc

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)

    roc_auc, pr_auc, best_mcc = evaluate_model(model, eval_loader)
    writer.add_scalar('Metrics/ROC_AUC', roc_auc, epoch)
    writer.add_scalar('Metrics/PR_AUC', pr_auc, epoch)
    writer.add_scalar('Metrics/MCC', best_mcc, epoch)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f} - ROC AUC: {roc_auc:.4f} - PR AUC: {pr_auc:.4f} - MCC: {best_mcc:.4f}")
    
    checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss / len(train_loader),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_mcc': best_mcc,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

writer.close()
torch.save(model.state_dict(), "protein_nn_model_final.pth")
