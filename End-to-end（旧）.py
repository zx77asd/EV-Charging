import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict

# Import optuna for hyperparameter optimization
try:
    import optuna
except ImportError:
    print("Warning: optuna not installed, using default hyperparameters")
    optuna = None

# Import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    VISUALIZATION_ENABLED = True
except ImportError as e:
    print(f"Warning: Visualization library import failed: {e}")
    print("Visualization functionality will be disabled")
    VISUALIZATION_ENABLED = False


# Implement simple scaler class to avoid dependency on scikit-learn
class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + 1e-10  # Avoid division by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_


# Implement evaluation metric functions
def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


# ============================================================
# CONFIG
# ============================================================
# Try to use CUDA if available
print("Checking CUDA availability...")
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test CUDA availability
    if DEVICE.type == "cuda":
        torch.tensor([1.0]).to(DEVICE)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()
    else:
        print("Using CPU device")
except Exception as e:
    print(f"CUDA initialization failed: {e}")
    print("Falling back to CPU device")
    DEVICE = torch.device("cpu")
NUM_SITES = 172
SEQ_LEN = 7
HORIZON = 1
LR = 1e-3
BATCH_SIZE = 16  # Reduced batch size to 32 to prevent CUDA OOM
EPOCHS = 50  # Set to 50 epochs as requested
EARLY_STOPPING_PATIENCE = 10

# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use absolute paths
PATH_TEMPORAL = r"D:\ai 绿色\temporal_features\features_temporal_features.csv"
PATH_WEATHER = r"D:\ai 绿色\temporal_features\features_weather_features.csv"
PATH_YDAY = r"D:\ai 绿色\temporal_features\features_yesterday_lags.csv"
PATH_LWEEK = r"D:\ai 绿色\temporal_features\features_last_week_lags.csv"
PATH_7AVG = r"D:\ai 绿色\temporal_features\features_7day_avg_lags.csv"
TARGET_FILE = r"D:\ai 绿色\temporal_features\features_date_targets.csv"

EDGE_CSV = r"D:\ai 绿色\spatial-embedding\hetero_edges_final.csv"
POI_TYPE_EMB_NPY = r"D:\ai 绿色\spatial-embedding\all_poi_embedding.npy"
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.pt")
SEED = 42

# ============================================================
# SET SEED
# ============================================================
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DATA LOADING
# ============================================================
def load_all_data():
    temporal_df = pd.read_csv(PATH_TEMPORAL).select_dtypes(include=[np.number])
    weather_df = pd.read_csv(PATH_WEATHER).select_dtypes(include=[np.number])
    yday_df = pd.read_csv(PATH_YDAY).select_dtypes(include=[np.number])
    lweek_df = pd.read_csv(PATH_LWEEK).select_dtypes(include=[np.number])
    avg7_df = pd.read_csv(PATH_7AVG).select_dtypes(include=[np.number])
    targets_df = pd.read_csv(TARGET_FILE)

    dates = pd.to_datetime(targets_df["date"])
    targets_df = targets_df.select_dtypes(include=[np.number])

    temporal = torch.tensor(temporal_df.values, dtype=torch.float32)
    weather = torch.tensor(weather_df.values, dtype=torch.float32)
    yday = torch.tensor(yday_df.values, dtype=torch.float32)
    lweek = torch.tensor(lweek_df.values, dtype=torch.float32)
    avg7 = torch.tensor(avg7_df.values, dtype=torch.float32)
    lags = torch.stack([yday, lweek, avg7], dim=-1)
    targets = torch.tensor(targets_df.values, dtype=torch.float32)

    return temporal, weather, lags, targets, dates


# ============================================================
# 🎯 新增: 数据预加载函数 (避免重复加载)
# ============================================================
# 【改动1】添加这个函数在load_all_data()之后
def load_data_once():
    """一次性加载所有数据,避免重复加载"""
    global _data_cache, _graph_cache

    print("\n" + "=" * 60)
    print("🔄 PRELOADING DATA (only once)")
    print("=" * 60)

    # 加载原始数据
    print("📊 Loading data files...")
    temporal, weather, lags, targets, dates = load_all_data()

    # 计算分割
    total_samples = (temporal.shape[0] - SEQ_LEN - HORIZON) // (SEQ_LEN + HORIZON) + 1
    train_samples = int(0.7 * total_samples)
    val_samples = int(0.15 * total_samples)
    test_samples = total_samples - train_samples - val_samples

    # 标准化
    print("⚙️  Standardizing data...")
    train_time_steps = train_samples * (SEQ_LEN + HORIZON)
    train_temporal = temporal[:train_time_steps + SEQ_LEN].numpy()
    train_weather = weather[:train_time_steps + SEQ_LEN].numpy()
    train_lags = lags[:train_time_steps + SEQ_LEN].numpy()
    train_targets = targets[SEQ_LEN:train_time_steps + SEQ_LEN + HORIZON - 1].numpy()

    (train_temporal, train_weather, train_lags, train_targets,
     t_scaler, w_scaler, lag_scaler, y_scaler) = fit_scalers(
        train_temporal, train_weather, train_lags, train_targets)

    temporal, weather, lags, targets = apply_scalers(
        temporal.numpy(), weather.numpy(), lags.numpy(), targets.numpy(),
        (t_scaler, w_scaler, lag_scaler, y_scaler))

    # 创建数据集
    print("📦 Creating dataset...")
    dataset = EVChargingDataset(temporal, weather, lags, targets, SEQ_LEN, HORIZON)

    # 缓存数据
    print("💾 Caching to memory...")
    _data_cache['processed_data'] = {
        'dataset': dataset,
        'train_idx': list(range(train_samples)),
        'val_idx': list(range(train_samples, train_samples + val_samples)),
        'test_idx': list(range(train_samples + val_samples, total_samples)),
        'scalers': (t_scaler, w_scaler, lag_scaler, y_scaler),
    }

    # 加载图
    print("🗺️  Loading graph...")
    graph = HeteroGraph(EDGE_CSV)
    poi_emb = torch.tensor(np.load(POI_TYPE_EMB_NPY), dtype=torch.float32)
    site_ids = list(range(NUM_SITES))
    poi_ids = list(range(NUM_SITES, NUM_SITES + poi_emb.shape[0]))

    # 缓存图
    print("💾 Caching graph...")
    _graph_cache['graph_data'] = {
        'graph': graph,
        'poi_emb': poi_emb,
        'site_ids': site_ids,
        'poi_ids': poi_ids
    }

    print("✅ DATA PRELOADED!")
    print("=" * 60 + "\n")


# ============================================================
# DATASET
# ============================================================
class EVChargingDataset(Dataset):
    def __init__(self, temporal, weather, lags, targets, seq_len, horizon):
        self.temporal = temporal
        self.weather = weather
        self.lags = lags
        self.targets = targets
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        # Calculate number of non-overlapping patches
        return (self.temporal.shape[0] - self.seq_len - self.horizon) // (self.seq_len + self.horizon) + 1

    def __getitem__(self, idx):
        # Calculate start index for non-overlapping patches
        start_idx = idx * (self.seq_len + self.horizon)
        t0, t1 = start_idx, start_idx + self.seq_len
        y0, y1 = start_idx + self.seq_len, start_idx + self.seq_len + self.horizon
        return (
            self.temporal[t0:t1],
            self.weather[t0:t1],
            self.lags[t0:t1],
            self.targets[y0:y1]
        )


# ============================================================
# SCALERS
# ============================================================
def fit_scalers(train_temporal, train_weather, train_lags, train_targets):
    t_scaler = SimpleScaler()
    w_scaler = SimpleScaler()
    lag_scaler = SimpleScaler()
    y_scaler = SimpleScaler()

    train_temporal = torch.tensor(t_scaler.fit_transform(train_temporal), dtype=torch.float32)
    train_weather = torch.tensor(w_scaler.fit_transform(train_weather), dtype=torch.float32)
    T, N, C = train_lags.shape
    lag_2d = train_lags.reshape(T * N, C)
    lag_scaled = lag_scaler.fit_transform(lag_2d).reshape(T, N, C)
    train_lags = torch.tensor(lag_scaled, dtype=torch.float32)
    train_targets = torch.tensor(y_scaler.fit_transform(train_targets.reshape(-1, N)), dtype=torch.float32).reshape(-1,
                                                                                                                    N)
    return train_temporal, train_weather, train_lags, train_targets, t_scaler, w_scaler, lag_scaler, y_scaler


def apply_scalers(temporal, weather, lags, targets, scalers):
    t_scaler, w_scaler, lag_scaler, y_scaler = scalers
    temporal = torch.tensor(t_scaler.transform(temporal), dtype=torch.float32)
    weather = torch.tensor(w_scaler.transform(weather), dtype=torch.float32)
    T, N, C = lags.shape
    lag_2d = lags.reshape(T * N, C)
    lag_scaled = lag_scaler.transform(lag_2d).reshape(T, N, C)
    lags = torch.tensor(lag_scaled, dtype=torch.float32)
    targets = torch.tensor(y_scaler.transform(targets.reshape(-1, N)), dtype=torch.float32).reshape(-1, N)
    return temporal, weather, lags, targets


# ============================================================
# HGT
# ============================================================
class HeteroGraph:
    def __init__(self, edge_csv):
        df = pd.read_csv(edge_csv)
        self.node_list = sorted(set(df["src"]).union(set(df["dst"])))
        self.num_nodes = len(self.node_list)
        self.rel2id = {}
        self.edge_index = defaultdict(list)
        for _, r in df.iterrows():
            s, d, rel = int(r["src"]), int(r["dst"]), r["edge_type"]
            if rel not in self.rel2id:
                self.rel2id[rel] = len(self.rel2id)
            self.edge_index[self.rel2id[rel]].append([s, d])
        for r in self.edge_index:
            self.edge_index[r] = torch.tensor(self.edge_index[r]).t().long()


class HGTLayer(nn.Module):
    def __init__(self, dim, num_rels, heads=4):
        super().__init__()
        self.dk = dim // heads
        self.heads = heads
        self.Wq = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_rels)])
        self.Wk = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_rels)])
        self.Wv = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_rels)])
        self.rel_att = nn.Parameter(torch.randn(num_rels, heads))
        self.out = nn.Linear(dim, dim)

    def forward(self, x, edge_index):
        out = torch.zeros_like(x)
        for r in edge_index:
            src, dst = edge_index[r]
            Q = self.Wq[r](x[dst]).view(-1, self.heads, self.dk)
            K = self.Wk[r](x[src]).view(-1, self.heads, self.dk)
            V = self.Wv[r](x[src]).view(-1, self.heads, self.dk)
            score = (Q * K).sum(-1) / (self.dk ** 0.5)
            score = score * self.rel_att[r]
            order = torch.argsort(dst)
            dst_sorted = dst[order]
            score_sorted = score[order]
            uniq, cnt = torch.unique_consecutive(dst_sorted, return_counts=True)
            alpha = torch.cat([torch.softmax(s, dim=0) for s in torch.split(score_sorted, cnt.tolist())], dim=0)
            msg = (alpha.unsqueeze(-1) * V[order]).reshape(-1, self.heads * self.dk)
            out.index_add_(0, dst_sorted, msg)
        return self.out(out)


class HGT(nn.Module):
    def __init__(self, graph, site_ids, poi_ids, poi_type_embedding, dim=64, num_layers=2, heads=4, device='cpu',
                 mlp_hidden_size=None, mlp_activation='relu'):
        super().__init__()
        self.device = device
        self.dim = dim
        self.graph = graph
        self.num_layers = num_layers
        self.heads = heads

        # All node mapping
        all_node_ids = sorted(graph.node_list)
        self.node2idx = {nid: i for i, nid in enumerate(all_node_ids)}
        self.idx2node = {i: nid for nid, i in self.node2idx.items()}
        self.num_nodes = len(all_node_ids)
        self.num_sites = len(site_ids)
        self.num_pois = len(poi_ids)

        # Create POI node set
        self.poi_ids_set = set(poi_ids)

        self.node_emb = nn.Embedding(self.num_nodes, dim).to(device)
        self.poi_features = nn.Parameter(poi_type_embedding.clone().detach().to(device), requires_grad=True)

        # Determine MLP hidden layer size
        if mlp_hidden_size is None:
            mlp_hidden_size = dim * 2

        # Select activation function
        activation_fn = nn.ReLU() if mlp_activation == 'relu' else nn.GELU()

        # MLP layer to transform static POI embedding to trainable dynamic embedding
        self.poi_feat_proj = nn.Sequential(
            nn.Linear(poi_type_embedding.shape[1], mlp_hidden_size),
            activation_fn,
            nn.Linear(mlp_hidden_size, dim)
        ).to(device)

        # Map edges
        self.edge_index = {}
        for etype, idx in graph.edge_index.items():
            mapped = torch.zeros_like(idx, dtype=torch.long)
            for i in range(idx.shape[0]):
                for j in range(idx.shape[1]):
                    node_id = idx[i, j].item()
                    mapped[i, j] = self.node2idx[node_id]
            self.edge_index[etype] = mapped.to(device)

        self.layers = nn.ModuleList(
            [HGTLayer(dim, num_rels=len(graph.rel2id), heads=heads).to(device) for _ in range(num_layers)])
        self.linear = nn.Linear(dim, dim).to(device)

        # Cache for spatial embeddings
        self._cached_emb = None

    def forward(self):
        # Check if we have a cached embedding and if parameters haven't changed
        if self._cached_emb is None:
            # Create a copy of node_emb.weight to avoid in-place modification of leaf variables
            x = self.node_emb.weight.clone()

            # Transform POI features through MLP and add to corresponding nodes
            poi_emb = self.poi_feat_proj(self.poi_features)
            for i, nid in enumerate(self.graph.node_list):
                if nid in self.poi_ids_set:
                    # Find corresponding POI index and ensure it's an integer
                    poi_idx = int(nid - self.num_sites)
                    if poi_idx >= 0 and poi_idx < self.num_pois:
                        x[i] = x[i] + poi_emb[poi_idx]

            for layer in self.layers:
                x = layer(x, self.edge_index)
            x = self.linear(x)

            # Cache the result
            self._cached_emb = x

        return self._cached_emb

    def clear_cache(self):
        """Clear the cached embedding to force recomputation"""
        self._cached_emb = None

    # ============================================================


# Temporal Transformer
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # Handle 4D input [B, T, N, D]
        if x.dim() == 4:
            # Expand positional encoding to [1, T, 1, D] to match input dimension
            pe = self.pe[:, :x.size(1), :].unsqueeze(2)
        else:
            # Handle 3D input [B, T, D]
            pe = self.pe[:, :x.size(1), :]
        return x + pe


class TemporalTransformer(nn.Module):
    def __init__(self, time_dim, weather_dim, lag_dim, d_model=64, nhead=4, num_layers=1, dropout=0.1, max_len=200):
        super().__init__()
        self.time_proj = nn.Sequential(nn.Linear(time_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
                                       nn.Dropout(dropout))
        self.weather_proj = nn.Sequential(nn.Linear(weather_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
                                          nn.Dropout(dropout))
        self.lag_proj = nn.Sequential(nn.Linear(lag_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
                                      nn.Dropout(dropout))
        self.gate_net = nn.Sequential(nn.Linear(3 * d_model, 3), nn.Sigmoid())
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
                                                   dropout=dropout, activation="gelu", batch_first=True,
                                                   norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal_feat, weather_feat, volume_lags):
        # Dimension transformation: temporal_feat and weather_feat are [B, T, C], volume_lags is [B, T, N, C]
        B, T, N, C_lag = volume_lags.shape

        # Project to the same dimension
        time_emb = self.time_proj(temporal_feat)  # [B, T, D]
        weather_emb = self.weather_proj(weather_feat)  # [B, T, D]
        lag_emb = self.lag_proj(volume_lags)  # [B, T, N, D]

        # Feature fusion: for each time step, fuse time and weather features, then broadcast to each site
        time_weather_emb = time_emb + weather_emb  # [B, T, D]
        time_weather_emb = time_weather_emb.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T, N, D]

        # Gated fusion
        combined = torch.cat([time_weather_emb, lag_emb, time_weather_emb * lag_emb], dim=-1)  # [B, T, N, 3D]
        gates = self.gate_net(combined)  # [B, T, N, 3]
        gates = gates.softmax(dim=-1)  # Ensure weights sum to 1

        # Weighted fusion
        fused_emb = gates[..., 0:1] * time_weather_emb + gates[..., 1:2] * lag_emb + gates[..., 2:3] * (
                time_weather_emb * lag_emb)
        fused_emb = self.dropout(fused_emb)

        # Positional encoding
        fused_emb = self.positional_encoding(fused_emb)

        # Transformer encoding
        # Reshape to [B*N, T, D] to fit Transformer
        fused_reshaped = fused_emb.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        encoded = self.transformer(fused_reshaped)
        # Reshape back to [B, T, N, D]
        encoded = encoded.reshape(B, N, T, -1).permute(0, 2, 1, 3)

        return encoded


# ============================================================
# Multi-Scale Feature Extraction
# ============================================================
class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self, in_dim, out_dim, scales):
        super().__init__()
        self.scales = scales
        self.conv_layers = nn.ModuleList()
        for scale in scales:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=(scale, 1), padding=(scale // 2, 0)),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_dim)
                )
            )
        self.fusion = nn.Linear(len(scales) * out_dim, out_dim)

    def forward(self, x):
        # x shape: (B, T, N, D)
        B, T, N, D = x.shape
        # Reshape for 2D convolution: (B, D, T, N)
        x_reshaped = x.permute(0, 3, 1, 2)
        # Apply different scale convolutions
        outputs = []
        for conv in self.conv_layers:
            out = conv(x_reshaped)
            # Reshape back: (B, T, N, D)
            out = out.permute(0, 2, 3, 1)
            outputs.append(out)
        # Concatenate along feature dimension
        combined = torch.cat(outputs, dim=-1)
        # Fuse features
        fused = self.fusion(combined)
        return fused


# ============================================================
# Cross-Attention Fusion
# ============================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads
        self.dropout = nn.Dropout(dropout)

        # Multi-head attention weights
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        # Bidirectional attention (time to space and space to time)
        self.spatial_to_temporal = nn.Linear(dim, dim)
        self.temporal_to_spatial = nn.Linear(dim, dim)

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.out = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, time_emb, spatial_emb):
        B, T, N, D = time_emb.shape

        # Check if spatial_emb has time dimension
        if spatial_emb.dim() == 4:
            # Spatial emb has time dimension [B, T, N, D]
            # Prepare query, key, value
            # Query comes from time features, key and value come from spatial features
            q = self.Wq(time_emb).view(B, T, N, self.heads, self.dk).permute(0, 3, 1, 2, 4)  # [B, H, T, N, dk]
            k = self.Wk(spatial_emb).view(B, T, N, self.heads, self.dk).permute(0, 3, 1, 2, 4)  # [B, H, T, N, dk]
            v = self.Wv(spatial_emb).view(B, T, N, self.heads, self.dk).permute(0, 3, 1, 2, 4)  # [B, H, T, N, dk]
        else:
            # Spatial emb has no time dimension [B, N, D]
            # Prepare query, key, value
            # Query comes from time features, key and value come from spatial features
            q = self.Wq(time_emb).view(B, T, N, self.heads, self.dk).permute(0, 3, 1, 2, 4)  # [B, H, T, N, dk]
            k = self.Wk(spatial_emb).view(B, N, self.heads, self.dk).unsqueeze(2).repeat(1, 1, T, 1, 1).permute(0, 3, 2, 1,
                                                                                                                4)  # [B, H, T, N, dk]
            v = self.Wv(spatial_emb).view(B, N, self.heads, self.dk).unsqueeze(2).repeat(1, 1, T, 1, 1).permute(0, 3, 2, 1,
                                                                                                                4)  # [B, H, T, N, dk]

        # Calculate attention scores
        score = torch.matmul(q, k.transpose(-2, -1)) / (self.dk ** 0.5)  # [B, H, T, N, N]
        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # [B, H, T, N, dk]
        out = out.permute(0, 2, 3, 1, 4).reshape(B, T, N, D)  # [B, T, N, D]

        # Bidirectional attention fusion
        if spatial_emb.dim() == 4:
            # Spatial emb has time dimension [B, T, N, D]
            spatial_context = self.spatial_to_temporal(spatial_emb)  # [B, T, N, D]
        else:
            # Spatial emb has no time dimension [B, N, D]
            spatial_context = self.spatial_to_temporal(spatial_emb).unsqueeze(1).repeat(1, T, 1, 1)  # [B, T, N, D]
        temporal_context = self.temporal_to_spatial(time_emb.mean(dim=1, keepdim=True)).repeat(1, T, 1,
                                                                                               1)  # [B, T, N, D]

        # Feature fusion
        fused_features = torch.cat([out, spatial_context, temporal_context], dim=-1)  # [B, T, N, 3D]
        fused_features = self.fusion(fused_features)

        # Residual connection and layer normalization
        out = self.layer_norm(fused_features + time_emb)
        return out


# ============================================================
# ATTENTION POOLING
# ============================================================
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        scores = self.attn(x).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        out = (x * weights).sum(dim=1)  # [B, D]
        return out


# ============================================================
# PATCH TST
# ============================================================
class PatchTST(nn.Module):
    def __init__(self, in_d_model, d_model=64, patch=7, nhead=4, num_layers=2, dim_ff=256, horizon=7):
        super().__init__()
        self.patch = patch
        self.d_model = d_model
        self.horizon = horizon
        self.dim_proj = nn.Linear(in_d_model, d_model)
        self.patch_proj = nn.Linear(patch * d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPooling(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        B, T, N, D = x.shape
        P = T // self.patch

        # First perform dimension transformation
        x = self.dim_proj(x)

        # Then perform patch processing
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        x = x.reshape(B * N, P, self.patch * x.shape[-1])
        x = self.patch_proj(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.head(x)
        return x.view(B, N, self.horizon).permute(0, 2, 1)


# ============================================================
# MULTI-SCALE FEATURE EXTRACTION
# ============================================================
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Create convolution layers with different receptive fields
        for i in range(num_scales):
            kernel_size = 2 * (i + 1) + 1  # 3, 5, 7...
            padding = kernel_size // 2
            self.convs.append(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.activations.append(nn.GELU())

        # Cross-scale feature fusion
        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (B, T, N, D)
        B, T, N, D = x.shape

        # Reshape to (B*N, D, T) to fit Conv1d
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * N, D, T)

        scale_features = []
        for i in range(self.num_scales):
            feat = self.convs[i](x_reshaped)  # (B*N, hidden_dim, T)
            feat = feat.permute(0, 2, 1)  # (B*N, T, hidden_dim)
            feat = self.norms[i](feat)
            feat = self.activations[i](feat)
            scale_features.append(feat)

        # Concatenate multi-scale features
        concatenated = torch.cat(scale_features, dim=-1)  # (B*N, T, hidden_dim*num_scales)
        fused = self.fusion(concatenated)  # (B*N, T, hidden_dim)
        fused = self.layer_norm(fused)

        # Reshape back to original shape
        fused = fused.reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B, T, N, hidden_dim)
        return fused


# ============================================================
# FULL MODEL
# ============================================================
class EVChargingModel(nn.Module):
    def __init__(self, time_dim, weather_dim, lag_dim, spatial_encoder,
                 temp_d_model=64, temp_nhead=4, temp_layers=1, temp_dropout=0.1,
                 fusion_heads=4, fusion_dropout=0.1,
                 patch_d_model=64, patch_patch=1, patch_nhead=4, patch_layers=2, patch_ff=256, patch_dropout=0.1,
                 horizon=3, weight_decay=0.01):
        super().__init__()
        self.temporal = TemporalTransformer(time_dim, weather_dim, lag_dim, d_model=temp_d_model, nhead=temp_nhead,
                                            num_layers=temp_layers, dropout=temp_dropout)
        self.spatial = spatial_encoder
        self.multi_scale = MultiScaleFeatureExtractor(input_dim=temp_d_model, hidden_dim=temp_d_model)
        self.fusion = CrossAttentionFusion(dim=temp_d_model, heads=fusion_heads, dropout=fusion_dropout)
        self.dropout = nn.Dropout(patch_dropout)
        self.head = PatchTST(in_d_model=temp_d_model, d_model=patch_d_model, patch=patch_patch, nhead=patch_nhead,
                             num_layers=patch_layers, dim_ff=patch_ff, horizon=horizon)
        self.weight_decay = weight_decay

    def __init__(self, time_dim, weather_dim, lag_dim, spatial_encoder,
                 temp_d_model=64, temp_nhead=4, temp_layers=1, temp_dropout=0.1,
                 fusion_heads=4, fusion_dropout=0.1,
                 patch_d_model=64, patch_patch=1, patch_nhead=4, patch_layers=2, patch_ff=256, patch_dropout=0.1,
                 horizon=3, weight_decay=0.01):
        super().__init__()
        self.temporal = TemporalTransformer(time_dim, weather_dim, lag_dim, d_model=temp_d_model, nhead=temp_nhead,
                                            num_layers=temp_layers, dropout=temp_dropout)
        self.multi_scale = MultiScaleFeatureExtraction(temp_d_model, temp_d_model, [2, 4])
        self.spatial = spatial_encoder
        self.fusion = CrossAttentionFusion(temp_d_model, heads=fusion_heads, dropout=fusion_dropout)
        self.dropout = nn.Dropout(fusion_dropout)
        self.head = PatchTST(in_d_model=temp_d_model, d_model=patch_d_model, patch=patch_patch, nhead=patch_nhead,
                             num_layers=patch_layers, dim_ff=patch_ff, horizon=horizon)
        self.weight_decay = weight_decay
        # Add a linear layer for time modulation
        self.time_modulation = nn.Linear(temp_d_model, temp_d_model)

    def forward(self, temporal_x, weather_x, lags_x):
        time_emb = self.temporal(temporal_x, weather_x, lags_x)
        # Add multi-scale feature extraction
        multi_scale_emb = self.multi_scale(time_emb)

        # Get batch size and number of sites from multi_scale_emb
        B, T, N, D = multi_scale_emb.shape

        # Get spatial embedding
        all_nodes_emb = self.spatial()
        # Directly extract first N nodes from HGT output as site embeddings
        # Assume first N nodes in HGT's node list are site nodes
        site_emb = all_nodes_emb[:N]
        # Ensure site embedding shape matches
        assert site_emb.shape == (N, D), f"Site embedding shape mismatch: {site_emb.shape} vs {(N, D)}"

        # Expand batch dimension
        site_emb = site_emb.unsqueeze(0).repeat(B, 1, 1)  # Shape: (B, N, D)

        # Ensure expanded site embedding shape is correct
        assert site_emb.shape == (B, N, D), f"Expanded site embedding shape mismatch: {site_emb.shape} vs {(B, N, D)}"

        # Method A: Add time-dependent modulation to spatial embedding
        # 1. Take mean across site dimension to get global temporal signal for each time step
        global_temporal = multi_scale_emb.mean(dim=2)  # Shape: (B, T, D)

        # 2. Apply linear layer for modulation
        modulated_time = self.time_modulation(global_temporal)  # Shape: (B, T, D)

        # 3. Expand to match site dimension and add to spatial embedding
        # Expand to (B, T, N, D)
        modulated_time_expanded = modulated_time.unsqueeze(2).repeat(1, 1, N, 1)
        # Expand site embedding to (B, T, N, D)
        site_emb_expanded = site_emb.unsqueeze(1).repeat(1, T, 1, 1)
        # Add modulated time to spatial embedding
        dynamic_site_emb = site_emb_expanded + modulated_time_expanded  # Shape: (B, T, N, D)

        # Now perform cross-attention fusion between temporal embedding and dynamic spatial embedding
        fused = self.fusion(multi_scale_emb, dynamic_site_emb)
        fused = self.dropout(fused)
        return self.head(fused)


# ============================================================
# METRICS & EARLY STOPPING
# ============================================================
def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


class EarlyStopping:
    def __init__(self, patience=10, mode="min"):
        self.patience, self.mode = patience, mode
        self.best, self.counter, self.early_stop = None, 0, False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False
        improvement = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improvement:
            self.best, self.counter = metric, 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


# ============================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================
# Global cache for data
_data_cache = {}

# Global cache for graph and POI embeddings
_graph_cache = {}


def objective(trial):
    # Get global DEVICE variable
    global DEVICE

    # ✅ 【改动2】objective函数开头改为直接从缓存获取数据
    global _data_cache, _graph_cache

    # ✅ 直接从缓存获取 (不再加载数据!)
    cached_data = _data_cache['processed_data']
    dataset = cached_data['dataset']
    train_idx = cached_data['train_idx']
    val_idx = cached_data['val_idx']
    t_scaler, w_scaler, lag_scaler, y_scaler = cached_data['scalers']

    cached_graph = _graph_cache['graph_data']
    graph = cached_graph['graph']
    poi_emb = cached_graph['poi_emb'].to(DEVICE)  # Move POI embeddings to the correct device
    site_ids = cached_graph['site_ids']
    poi_ids = cached_graph['poi_ids']

    # ❌ 删除这段:
    # if cache_key not in _data_cache:
    #     print("Loading and processing data...")
    #     ...

    # ❌ 删除这段:
    # if graph_cache_key not in _graph_cache:
    #     print("Loading graph...")
    #     ...

    # Hyperparameter search space (slightly reduced range)
    batch_size = 16  # Set batch size to 16 as requested
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)  # Reduced learning rate range
    temp_d_model = trial.suggest_categorical('temp_d_model', [64, 128])  # Slightly reduced model dimension range
    temp_nhead = trial.suggest_categorical('temp_nhead', [2, 4])  # Slightly reduced attention heads range
    temp_layers = trial.suggest_int('temp_layers', low=1, high=2, step=1)  # Slightly reduced Transformer layers range
    temp_dropout = trial.suggest_float('temp_dropout', 0.0, 0.1)  # Slightly reduced dropout range
    fusion_heads = trial.suggest_categorical('fusion_heads',
                                             [2, 4, 8])  # Slightly reduced fusion attention heads range
    fusion_dropout = trial.suggest_float('fusion_dropout', 0.0, 0.1)  # Slightly reduced dropout range
    patch_d_model = trial.suggest_categorical('patch_d_model',
                                              [64, 128])  # Slightly reduced model dimension range
    patch_nhead = trial.suggest_categorical('patch_nhead', [2, 4])  # Slightly reduced attention heads range
    patch_layers = trial.suggest_int('patch_layers', low=1, high=2, step=1)  # Slightly reduced Transformer layers range
    patch_ff = trial.suggest_categorical('patch_ff',
                                         [128, 256])  # Slightly reduced feed-forward network dimension range
    patch_dropout = trial.suggest_float('patch_dropout', 0.0, 0.1)  # Slightly reduced dropout range
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)  # Reduced weight decay range
    # Label smoothing parameter
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1)  # Slightly reduced label smoothing range
    # MLP layer hyperparameters
    mlp_hidden_size = trial.suggest_categorical('mlp_hidden_size',
                                                [64, 128])  # Slightly reduced MLP hidden layer size range
    mlp_activation = trial.suggest_categorical('mlp_activation',
                                               ['relu', 'gelu'])  # Slightly reduced activation function options

    # Create DataLoader
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size)

    # Clear CUDA cache before loading graph data to free up memory
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # Initialize HGT model
    try:
        # Initialize HGT with device parameter, it will handle device placement internally
        hgt_model = HGT(
            graph, site_ids, poi_ids, poi_emb,
            dim=temp_d_model, device=DEVICE,
            mlp_hidden_size=mlp_hidden_size,
            mlp_activation=mlp_activation
        )
    except torch.OutOfMemoryError:
        print("CUDA out of memory when initializing HGT. Falling back to CPU...")
        DEVICE = torch.device('cpu')
        hgt_model = HGT(
            graph, site_ids, poi_ids, poi_emb,
            dim=temp_d_model, device=DEVICE,
            mlp_hidden_size=mlp_hidden_size,
            mlp_activation=mlp_activation
        )

    # Initialize EVChargingModel
    model = EVChargingModel(
        time_dim=dataset.temporal.shape[-1],
        weather_dim=dataset.weather.shape[-1],
        lag_dim=dataset.lags.shape[-1],
        spatial_encoder=hgt_model,
        temp_d_model=temp_d_model,
        temp_nhead=temp_nhead,
        temp_layers=temp_layers,
        temp_dropout=temp_dropout,
        fusion_heads=fusion_heads,
        fusion_dropout=fusion_dropout,
        patch_d_model=patch_d_model,
        patch_patch=1,
        patch_nhead=patch_nhead,
        patch_layers=patch_layers,
        patch_ff=patch_ff,
        patch_dropout=patch_dropout,
        horizon=HORIZON,
        weight_decay=weight_decay
    ).to(DEVICE)  # Move model to the correct device

    # Move model to device with memory management
    try:
        # Move components to device in smaller steps
        model.temporal = model.temporal.to(DEVICE)
        model.multi_scale = model.multi_scale.to(DEVICE)
        model.fusion = model.fusion.to(DEVICE)
        model.dropout = model.dropout.to(DEVICE)
        model.head = model.head.to(DEVICE)
    except torch.OutOfMemoryError:
        print("CUDA out of memory when initializing EVChargingModel. Falling back to CPU...")
        DEVICE = torch.device('cpu')
        model = EVChargingModel(
            time_dim=dataset.temporal.shape[-1],
            weather_dim=dataset.weather.shape[-1],
            lag_dim=dataset.lags.shape[-1],
            spatial_encoder=hgt_model,
            temp_d_model=temp_d_model,
            temp_nhead=temp_nhead,
            temp_layers=temp_layers,
            temp_dropout=temp_dropout,
            fusion_heads=fusion_heads,
            fusion_dropout=fusion_dropout,
            patch_d_model=patch_d_model,
            patch_patch=1,
            patch_nhead=patch_nhead,
            patch_layers=patch_layers,
            patch_ff=patch_ff,
            patch_dropout=patch_dropout,
            horizon=HORIZON,
            weight_decay=weight_decay
        )

    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Add label smoothing MSE loss
    class LabelSmoothingMSELoss(nn.Module):
        def __init__(self, smoothing=0.0):
            super().__init__()
            self.smoothing = smoothing
            self.mse = nn.MSELoss()

        def forward(self, pred, target):
            if self.smoothing > 0:
                # Add smoothing to target values
                target = target * (1 - self.smoothing) + torch.mean(target, dim=1, keepdim=True) * self.smoothing
            return self.mse(pred, target)

    loss_fn = LabelSmoothingMSELoss(smoothing=label_smoothing)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="min")

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    # Train model with early stopping
    best_val_loss = float("inf")
    for epoch in range(50):  # Set to 50 epochs with early stopping
        model.train()
        train_losses = []
        for x1, x2, x3, y in train_loader:
            try:
                optimizer.zero_grad()
                # Move data to device with error handling
                try:
                    x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
                except Exception as e:
                    print(f"Error moving data to device: {e}")
                    print("Falling back to CPU...")
                    DEVICE = torch.device('cpu')
                    x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
                    # Reinitialize model on CPU
                    model = EVChargingModel(
                        time_dim=dataset.temporal.shape[-1],
                        weather_dim=dataset.weather.shape[-1],
                        lag_dim=dataset.lags.shape[-1],
                        spatial_encoder=hgt_model,
                        temp_d_model=temp_d_model,
                        temp_nhead=temp_nhead,
                        temp_layers=temp_layers,
                        temp_dropout=temp_dropout,
                        fusion_heads=fusion_heads,
                        fusion_dropout=fusion_dropout,
                        patch_d_model=patch_d_model,
                        patch_patch=1,
                        patch_nhead=patch_nhead,
                        patch_layers=patch_layers,
                        patch_ff=patch_ff,
                        patch_dropout=patch_dropout,
                        horizon=HORIZON,
                        weight_decay=weight_decay
                    )
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

                # Disable mixed precision training to avoid type mismatch issues
                pred = model(x1, x2, x3)
                loss = loss_fn(pred, y)

                # Backward pass with error handling
                try:
                    loss.backward()
                except Exception as e:
                    print(f"Error during backward pass: {e}")
                    print("Falling back to CPU...")
                    DEVICE = torch.device('cpu')
                    # Reinitialize model on CPU
                    model = EVChargingModel(
                        time_dim=dataset.temporal.shape[-1],
                        weather_dim=dataset.weather.shape[-1],
                        lag_dim=dataset.lags.shape[-1],
                        spatial_encoder=hgt_model,
                        temp_d_model=temp_d_model,
                        temp_nhead=temp_nhead,
                        temp_layers=temp_layers,
                        temp_dropout=temp_dropout,
                        fusion_heads=fusion_heads,
                        fusion_dropout=fusion_dropout,
                        patch_d_model=patch_d_model,
                        patch_patch=1,
                        patch_nhead=patch_nhead,
                        patch_layers=patch_layers,
                        patch_ff=patch_ff,
                        patch_dropout=patch_dropout,
                        horizon=HORIZON,
                        weight_decay=weight_decay
                    )
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                    # Continue to next batch
                    continue

                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                # Clear HGT cache after parameter update
                model.spatial.clear_cache()
                train_losses.append(loss.item())

                # Clear CUDA cache after each batch to prevent memory accumulation
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error during training batch: {e}")
                print("Falling back to CPU...")
                DEVICE = torch.device('cpu')
                # Reinitialize HGT model on CPU
                hgt_model = HGT(
                    graph, site_ids, poi_ids, poi_emb,
                    dim=temp_d_model, device=DEVICE,
                    mlp_hidden_size=mlp_hidden_size,
                    mlp_activation=mlp_activation
                )
                # Reinitialize model on CPU
                model = EVChargingModel(
                    time_dim=dataset.temporal.shape[-1],
                    weather_dim=dataset.weather.shape[-1],
                    lag_dim=dataset.lags.shape[-1],
                    spatial_encoder=hgt_model,
                    temp_d_model=temp_d_model,
                    temp_nhead=temp_nhead,
                    temp_layers=temp_layers,
                    temp_dropout=temp_dropout,
                    fusion_heads=fusion_heads,
                    fusion_dropout=fusion_dropout,
                    patch_d_model=patch_d_model,
                    patch_patch=1,
                    patch_nhead=patch_nhead,
                    patch_layers=patch_layers,
                    patch_ff=patch_ff,
                    patch_dropout=patch_dropout,
                    horizon=HORIZON,
                    weight_decay=weight_decay
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                # Continue to next batch
                continue
        train_loss = np.mean(train_losses) if train_losses else float('inf')

        # Update learning rate
        scheduler.step()

        # Clear CUDA cache to free up memory
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # Perform validation every 2 epochs to save time
        if epoch % 2 == 0:
            model.eval()
            val_losses, val_trues, val_preds = [], [], []
            with torch.no_grad():
                for x1, x2, x3, y in val_loader:
                    x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
                    pred = model(x1, x2, x3)
                    val_losses.append(loss_fn(pred, y).item())
                    val_trues.append(y.cpu().numpy())
                    val_preds.append(pred.cpu().numpy())
            val_loss = np.mean(val_losses)

            # Clear CUDA cache after validation
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            # Early stopping
            if early_stopper(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        else:
            # Use the last validation loss for early stopping
            val_loss = best_val_loss

    return best_val_loss


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # ✅ 【改动3】在main函数最开始添加预加载
    print("🔄 Starting data preloading...")
    load_data_once()  # 耗时5-10秒,但之后的50个trial都快10倍!

    # Check if using optuna for hyperparameter optimization
    if optuna is not None and True:  # Set to True to enable hyperparameter optimization
        print("Starting hyperparameter optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=25)  # Set to 25 trials as requested

        print("Best hyperparameters:")
        print(study.best_params)
        print("Best validation loss:")
        print(study.best_value)

        # Retrain with best hyperparameters
        best_params = study.best_params
        BATCH_SIZE = 64  # Fix batch_size to 1024
        LR = best_params['lr']
        temp_d_model = best_params['temp_d_model']
        temp_nhead = best_params['temp_nhead']
        temp_layers = best_params['temp_layers']
        temp_dropout = best_params['temp_dropout']
        fusion_heads = best_params['fusion_heads']
        fusion_dropout = best_params['fusion_dropout']
        patch_d_model = best_params['patch_d_model']
        patch_nhead = best_params['patch_nhead']
        patch_layers = best_params['patch_layers']
        patch_ff = best_params['patch_ff']
        patch_dropout = best_params['patch_dropout']
        weight_decay = best_params['weight_decay']
        # MLP hyperparameters
        mlp_hidden_size = best_params.get('mlp_hidden_size', temp_d_model * 2)
        mlp_activation = best_params.get('mlp_activation', 'relu')
    else:
        # Use default hyperparameters
        print("Using default hyperparameters...")
        BATCH_SIZE = 16
        LR = 1e-3
        temp_d_model = 64
        temp_nhead = 4
        temp_layers = 2
        temp_dropout = 0.1
        fusion_heads = 4
        fusion_dropout = 0.1
        patch_d_model = 64
        patch_nhead = 4
        patch_layers = 2
        patch_ff = 256
        patch_dropout = 0.1
        weight_decay = 1e-4
        # MLP default hyperparameters
        mlp_hidden_size = temp_d_model * 2
        mlp_activation = 'relu'

    # 使用预加载的数据
    cached_data = _data_cache['processed_data']
    dataset = cached_data['dataset']
    train_idx = cached_data['train_idx']
    val_idx = cached_data['val_idx']
    test_idx = cached_data['test_idx']
    t_scaler, w_scaler, lag_scaler, y_scaler = cached_data['scalers']

    cached_graph = _graph_cache['graph_data']
    graph = cached_graph['graph']
    poi_emb = cached_graph['poi_emb']
    site_ids = cached_graph['site_ids']
    poi_ids = cached_graph['poi_ids']

    train_loader = DataLoader(Subset(dataset, train_idx), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), BATCH_SIZE)
    test_loader = DataLoader(Subset(dataset, test_idx), BATCH_SIZE)

    # HGT
    hgt_model = HGT(
        graph, site_ids, poi_ids, poi_emb,
        dim=temp_d_model, device=DEVICE,
        mlp_hidden_size=mlp_hidden_size,
        mlp_activation=mlp_activation
    ).to(DEVICE)

    model = EVChargingModel(
        time_dim=dataset.temporal.shape[-1],
        weather_dim=dataset.weather.shape[-1],
        lag_dim=dataset.lags.shape[-1],
        spatial_encoder=hgt_model,
        temp_d_model=temp_d_model,
        temp_nhead=temp_nhead,
        temp_layers=temp_layers,
        temp_dropout=temp_dropout,
        fusion_heads=fusion_heads,
        fusion_dropout=fusion_dropout,
        patch_d_model=patch_d_model,
        patch_patch=1,
        patch_nhead=patch_nhead,
        patch_layers=patch_layers,
        patch_ff=patch_ff,
        patch_dropout=patch_dropout,
        horizon=HORIZON,
        weight_decay=weight_decay
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="min")

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # Initialize training history
    train_loss_history = []
    val_loss_history = []
    val_r2_history = []

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for x1, x2, x3, y in train_loader:
            optimizer.zero_grad()
            x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
            pred = model(x1, x2, x3)
            loss = loss_fn(pred, y)
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            # Clear HGT cache after parameter update
            model.spatial.clear_cache()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # Update learning rate
        scheduler.step()
        train_loss_history.append(train_loss)

        # Clear CUDA cache to free up memory
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # Perform validation every 2 epochs to save time
        if epoch % 2 == 0:
            model.eval()
            val_losses, val_trues, val_preds = [], [], []
            with torch.no_grad():
                for x1, x2, x3, y in val_loader:
                    x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
                    pred = model(x1, x2, x3)
                    val_losses.append(loss_fn(pred, y).item())
                    val_trues.append(y.cpu().numpy())
                    val_preds.append(pred.cpu().numpy())
            val_loss = np.mean(val_losses)
            val_loss_history.append(val_loss)

            # Clear CUDA cache after validation
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            val_trues = np.concatenate(val_trues, 0)
            val_preds = np.concatenate(val_preds, 0)
        else:
            # Use the last validation loss and predictions for consistency
            val_loss = val_loss_history[-1] if val_loss_history else float('inf')
            val_loss_history.append(val_loss)
            # Use dummy values for val_trues and val_preds
            val_trues = np.array([])
            val_preds = np.array([])

        # Reshape to 2D array for inverse transformation
        if val_trues.size > 0 and val_preds.size > 0:
            val_trues_shape = val_trues.shape
            val_preds_shape = val_preds.shape

            val_trues_2d = val_trues.reshape(-1, val_trues_shape[-1])
            val_preds_2d = val_preds.reshape(-1, val_preds_shape[-1])

            val_trues_inv = y_scaler.inverse_transform(val_trues_2d).reshape(val_trues_shape)

            val_preds_inv = y_scaler.inverse_transform(val_preds_2d).reshape(val_preds_shape)
            metrics = compute_metrics(val_trues_inv, val_preds_inv)
            val_r2_history.append(metrics[0])
        else:
            # Use the last metrics for consistency
            if val_r2_history:
                metrics = (val_r2_history[-1], 0, 0)  # Use last R2, dummy RMSE and MAE
            else:
                metrics = (0, 0, 0)  # Dummy metrics for first epoch
            val_r2_history.append(metrics[0])

        # Print full metrics
        print(f"Epoch {epoch + 1} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f} | "
              f"R2 {metrics[0]:.4f} | RMSE {metrics[1]:.4f} | MAE {metrics[2]:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch + 1, "model_state": model.state_dict()}, CHECKPOINT_FILE)
        if early_stopper(val_loss):
            print("Early stopping triggered.")
            break

    # TEST
    ckpt = torch.load(CHECKPOINT_FILE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    test_losses, test_trues, test_preds = [], [], []
    with torch.no_grad():
        for x1, x2, x3, y in test_loader:
            x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
            pred = model(x1, x2, x3)
            test_losses.append(loss_fn(pred, y).item())
            test_trues.append(y.cpu().numpy())
            test_preds.append(pred.cpu().numpy())
    test_trues = np.concatenate(test_trues, 0)
    test_preds = np.concatenate(test_preds, 0)

    # Reshape to 2D array for inverse transformation
    test_trues_shape = test_trues.shape
    test_preds_shape = test_preds.shape

    test_trues_2d = test_trues.reshape(-1, test_trues_shape[-1])
    test_preds_2d = test_preds.reshape(-1, test_preds_shape[-1])

    test_trues_inv = y_scaler.inverse_transform(test_trues_2d).reshape(test_trues_shape)
    test_preds_inv = y_scaler.inverse_transform(test_preds_2d).reshape(test_preds_shape)
    metrics_test = compute_metrics(test_trues_inv, test_preds_inv)

    # Print full test metrics
    print(f"Test Loss {np.mean(test_losses):.6f} | "
          f"Test R2 {metrics_test[0]:.4f} | "
          f"Test RMSE {metrics_test[1]:.4f} | "
          f"Test MAE {metrics_test[2]:.4f}")

    # Save test results
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics to file (including best parameters)
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Test Metrics\n")
        f.write(f"Loss: {np.mean(test_losses):.6f}\n")
        f.write(f"R2: {metrics_test[0]:.4f}\n")
        f.write(f"RMSE: {metrics_test[1]:.4f}\n")
        f.write(f"MAE: {metrics_test[2]:.4f}\n\n")

        f.write("Best Hyperparameters\n")
        if optuna is not None and 'best_params' in locals():
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write("Using default hyperparameters\n")
    print(f"Test metrics saved to: {metrics_path}")


