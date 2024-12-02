import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, accuracy_score, f1_score, \
    precision_score, recall_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

# 1. 数据加载与预处理
file_path = r"D:\Sepsis\risk prediction model\fs\10genesexp.xlsx"
data = pd.read_excel(file_path)

samples = data.iloc[:, 0]
labels = data.iloc[:, 1]
features = data.iloc[:, 2:]

# 数据标准化
scaler = RobustScaler()
features_scaled = scaler.fit_transform(features)

# 数据增强（SMOTE）
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_scaled, labels)

# 训练/测试划分
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# PyTorch张量转换
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # 增加批量大小
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# 2. 定义深度学习模型
class AdvancedOptimizedModel(nn.Module):
    def __init__(self, input_dim):
        super(AdvancedOptimizedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.6)  # 增强正则化

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = x.unsqueeze(1)  # 增加时间维度
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)  # 移除时间维度
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        return self.fc5(x)


# 3. 使用 Focal Loss 作为损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-bce_loss)  # pt = exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# 4. 定义早停策略
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# 5. 训练与评估函数
def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, epochs, early_stopping):
    model.train()
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))

        # 打印损失
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

        # 动态调整学习率（传入验证集损失）
        scheduler.step(test_losses[-1])

        # 早停
        early_stopping(test_losses[-1])
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return train_losses, test_losses


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            y_probs.extend(probabilities.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    return y_true, y_probs


# 计算 AUROC 置信区间
def bootstrap_auc_ci(y_true, y_probs, n_iterations=1000, ci_percentile=95):
    auc_scores = []
    for _ in range(n_iterations):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_resampled = y_true[indices]
        y_probs_resampled = y_probs[indices]
        auc_score = roc_auc_score(y_true_resampled, y_probs_resampled)
        auc_scores.append(auc_score)

    auc_scores = np.array(auc_scores)
    lower_bound = np.percentile(auc_scores, (100 - ci_percentile) / 2)
    upper_bound = np.percentile(auc_scores, 100 - (100 - ci_percentile) / 2)
    return lower_bound, upper_bound


# 计算 AUPRC 置信区间
def bootstrap_prc_ci(y_true, y_probs, n_iterations=1000, ci_percentile=95):
    pr_auc_scores = []
    for _ in range(n_iterations):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_resampled = y_true[indices]
        y_probs_resampled = y_probs[indices]

        precision, recall, _ = precision_recall_curve(y_true_resampled, y_probs_resampled)
        pr_auc = auc(recall, precision)
        pr_auc_scores.append(pr_auc)

    pr_auc_scores = np.array(pr_auc_scores)
    lower_bound = np.percentile(pr_auc_scores, (100 - ci_percentile) / 2)
    upper_bound = np.percentile(pr_auc_scores, 100 - (100 - ci_percentile) / 2)
    return lower_bound, upper_bound


# 6. 模型初始化与训练
input_dim = X_train.shape[1]
model = AdvancedOptimizedModel(input_dim)

criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
early_stopping = EarlyStopping(patience=10, delta=0.001)

epochs = 100
train_losses, test_losses = train_model(
    model, criterion, optimizer, scheduler, train_loader, test_loader, epochs, early_stopping
)

# 可视化损失曲线
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs. Testing Loss")
plt.show()

# 7. 模型评估
y_true, y_probs = evaluate_model(model, test_loader)

# 计算 AUROC 和 AUPRC 及其置信区间
roc_auc = roc_auc_score(y_true, y_probs)
roc_auc_lower, roc_auc_upper = bootstrap_auc_ci(y_true, y_probs)

precision, recall, _ = precision_recall_curve(y_true, y_probs)
pr_auc = auc(recall, precision)
pr_auc_lower, pr_auc_upper = bootstrap_prc_ci(y_true, y_probs)

# 计算 Accuracy, F1 Score, Precision 和 Recall
accuracy = accuracy_score(y_true, (y_probs >= 0.5).astype(int))
f1 = f1_score(y_true, (y_probs >= 0.5).astype(int))
precision_score_val = precision_score(y_true, (y_probs >= 0.5).astype(int))
recall_score_val = recall_score(y_true, (y_probs >= 0.5).astype(int))

# 输出结果
print(f"AUROC: {roc_auc:.4f} (95% CI: {roc_auc_lower:.4f}, {roc_auc_upper:.4f})")
print(f"AUPRC: {pr_auc:.4f} (95% CI: {pr_auc_lower:.4f}, {pr_auc_upper:.4f})")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision_score_val:.4f}")
print(f"Recall: {recall_score_val:.4f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_true, y_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 绘制PR曲线
plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 混淆矩阵
cm = confusion_matrix(y_true, (y_probs >= 0.5).astype(int))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
