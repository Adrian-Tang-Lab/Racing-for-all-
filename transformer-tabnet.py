import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, roc_curve
from imblearn.combine import SMOTEENN
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.utils import resample

# 数据加载和预处理
file_path = r"D:\Sepsis\risk prediction model\fs\10genesexp.xlsx"
data = pd.read_excel(file_path)
samples = data.iloc[:, 0]
labels = data.iloc[:, 1]
features = data.iloc[:, 2:]

scaler = RobustScaler()
features_scaled = scaler.fit_transform(features)

smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(features_scaled, labels)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads=8, n_layers=3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=128, nhead=n_heads, dim_feedforward=256, dropout=0.5, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.fc = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        return self.fc(x)


# TabNet 模型
class TabNetClassifierWrapper:
    def __init__(self, input_dim):
        self.model = TabNetClassifier(
            input_dim=input_dim,
            output_dim=2,
            n_d=128,
            n_a=128,
            n_steps=8,
            gamma=1.3,
            lambda_sparse=1e-5,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=0.0003),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
        )

    def train(self, X_train, y_train, X_valid, y_valid, max_epochs=300):
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=["valid"],
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=30,
            batch_size=64,
            virtual_batch_size=32,
            num_workers=0,
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# 初始化 Transformer 模型
transformer_model = TransformerModel(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(transformer_model.parameters(), lr=0.0003, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


# 定义评估函数，包含 95% CI 的计算
def evaluate_model_with_ci(y_true, y_probs, n_bootstrap=1000, random_state=42):
    # 计算分类指标
    y_pred = (y_probs >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)

    # 计算 AUROC 和 AUPRC
    roc_auc = roc_auc_score(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    # Bootstrap 方法计算 95% CI
    rng = np.random.default_rng(random_state)
    roc_auc_scores = []
    pr_auc_scores = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip if only one class is present in the sample
        roc_auc_scores.append(roc_auc_score(y_true[indices], y_probs[indices]))
        precision, recall, _ = precision_recall_curve(y_true[indices], y_probs[indices])
        pr_auc_scores.append(auc(recall, precision))

    roc_auc_ci = np.percentile(roc_auc_scores, [2.5, 97.5])
    pr_auc_ci = np.percentile(pr_auc_scores, [2.5, 97.5])

    return {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1-score": report["1"]["f1-score"],
        "roc_auc": roc_auc,
        "roc_auc_ci": roc_auc_ci,
        "pr_auc": pr_auc,
        "pr_auc_ci": pr_auc_ci,
    }


# Transformer 模型训练和评估
transformer_train_loss = []
transformer_valid_loss = []
transformer_y_true, transformer_y_probs = [], []

transformer_model.train()
for epoch in range(150):
    transformer_model.train()
    epoch_loss = 0.0
    valid_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = transformer_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 在每个epoch结束后进行验证
    transformer_model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = transformer_model(X_batch)
            loss = criterion(outputs, y_batch)
            valid_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            transformer_y_probs.extend(probabilities.cpu().numpy())
            transformer_y_true.extend(y_batch.cpu().numpy())

    transformer_train_loss.append(epoch_loss / len(train_loader))
    transformer_valid_loss.append(valid_loss / len(test_loader))

    print(
        f"Epoch {epoch + 1}/150, Train Loss: {epoch_loss / len(train_loader):.4f}, Validation Loss: {valid_loss / len(test_loader):.4f}")

# TabNet 模型训练和评估
tabnet_model = TabNetClassifierWrapper(input_dim=X_train.shape[1])
tabnet_model.train(
    X_train=X_train,
    y_train=y_train.to_numpy(),
    X_valid=X_test,
    y_valid=y_test.to_numpy(),
    max_epochs=300,
)

# TabNet 模型评估
tabnet_y_probs = tabnet_model.predict_proba(X_test)[:, 1]
tabnet_y_true = y_test.to_numpy()

# 评估 Transformer 模型
transformer_metrics = evaluate_model_with_ci(
    np.array(transformer_y_true), np.array(transformer_y_probs)
)

# 评估 TabNet 模型
tabnet_metrics = evaluate_model_with_ci(
    np.array(tabnet_y_true), np.array(tabnet_y_probs)
)

# 打印结果
print("\nTransformer Metrics:")
for key, value in transformer_metrics.items():
    if isinstance(value, (list, np.ndarray)):
        print(f"{key}: {value[0]:.4f} - {value[1]:.4f} (95% CI)")
    else:
        print(f"{key}: {value:.4f}")

print("\nTabNet Metrics:")
for key, value in tabnet_metrics.items():
    if isinstance(value, (list, np.ndarray)):
        print(f"{key}: {value[0]:.4f} - {value[1]:.4f} (95% CI)")
    else:
        print(f"{key}: {value:.4f}")

# 检查是否存在过拟合
import matplotlib.pyplot as plt

plt.plot(range(1, 151), transformer_train_loss, label='Train Loss')
plt.plot(range(1, 151), transformer_valid_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Transformer Model: Train and Validation Loss')
plt.show()
