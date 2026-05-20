# 除了没有revin外，它相较于end-to-end(旧）代码而言，还改动了一些小地方，但总体结构没有发生变化
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False

# 标准化工具
class SiteAwareScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.scale_ = np.std(X, axis=0, keepdims=True) + 1e-10
        return self

    def transform(self, X): return (X - self.mean_) / self.scale_

    def fit_transform(self, X): return self.fit(X).transform(X)
class DemandScaler:
    """仅使用训练时间段拟合的逐站点 z-score 标准化器。"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.scale_ = np.std(X, axis=0, keepdims=True) + 1e-10
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_

# 指标函数
def r2_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))


def mean_absolute_error(y_true, y_pred): return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def mean_squared_error(y_true, y_pred): return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def compute_metrics(y_true, y_pred):
    """计算全局指标和逐站点 R2。"""
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    global_r2 = r2_score(y_true.flatten(), y_pred.flatten())

    site_r2s = []
    for i in range(y_true.shape[1]):
        if np.var(y_true[:, i]) > 1e-5:
            site_r2s.append(r2_score(y_true[:, i], y_pred[:, i]))

    mean_site_r2 = np.mean(site_r2s) if site_r2s else 0.0
    median_site_r2 = np.median(site_r2s) if site_r2s else 0.0

    return global_r2, mean_site_r2, median_site_r2, rmse, mae


def compute_horizon_metrics(y_true, y_pred):
    if y_true.ndim != 3:
        raise ValueError("Expected y_true and y_pred with shape [samples, horizon, sites].")
    horizon_metrics = []
    for h in range(y_true.shape[1]):
        global_r2, mean_r2, median_r2, rmse, mae = compute_metrics(y_true[:, h, :], y_pred[:, h, :])
        horizon_metrics.append({
            "horizon": h + 1,
            "global_r2": global_r2,
            "avg_site_r2": mean_r2,
            "median_site_r2": median_r2,
            "rmse": rmse,
            "mae": mae,
        })
    return horizon_metrics


def format_horizon_metrics(horizon_metrics):
    return " | ".join(
        f"H{m['horizon']}: R2 {m['global_r2']:.4f}, AvgSiteR2 {m['avg_site_r2']:.4f}, "
        f"RMSE {m['rmse']:.4f}, MAE {m['mae']:.4f}"
        for m in horizon_metrics
    )


def make_purged_time_splits(total_samples, seq_len, horizon, train_ratio=0.7, val_ratio=0.15):
    train_count = int(train_ratio * total_samples)
    val_count = int(val_ratio * total_samples)

    # Rolling windows near a split boundary can share target days. Purging
    # horizon - 1 samples keeps labels disjoint across train/val/test.
    purge_gap = max(0, horizon - 1)

    train_idx = list(range(0, train_count))
    val_start = train_count + purge_gap
    val_end = min(val_start + val_count, total_samples)
    test_start = val_end + purge_gap

    val_idx = list(range(val_start, val_end))
    test_idx = list(range(test_start, total_samples))

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            f"Empty split after purging: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    return train_idx, val_idx, test_idx, purge_gap

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SITES = 172
SEQ_LEN = 7

HORIZON = 3

BATCH_SIZE = 16
EPOCHS = int(os.environ.get("NO_REVIN_EPOCHS", "50"))
OPTUNA_TRIALS = int(os.environ.get("NO_REVIN_OPTUNA_TRIALS", "50"))
OPTUNA_EPOCHS = int(os.environ.get("NO_REVIN_OPTUNA_EPOCHS", "12"))
RUN_OPTUNA = os.environ.get("NO_REVIN_RUN_OPTUNA", "1") != "0"
EARLY_STOPPING_PATIENCE = 15
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.deterministic = not PIN_MEMORY
torch.backends.cudnn.benchmark = PIN_MEMORY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "result_no_revin")
PATH_TEMPORAL = r"D:\ai 绿色\temporal_features\features_temporal_features.csv"
PATH_WEATHER = r"D:\ai 绿色\temporal_features\features_weather_features.csv"
PATH_YDAY = r"D:\ai 绿色\temporal_features\features_yesterday_lags.csv"
PATH_LWEEK = r"D:\ai 绿色\temporal_features\features_last_week_lags.csv"
PATH_7AVG = r"D:\ai 绿色\temporal_features\features_7day_avg_lags.csv"
TARGET_FILE = r"D:\ai 绿色\temporal_features\features_date_targets.csv"

EDGE_CSV = r"D:\ai 绿色\spatial-embedding\hetero_edges_final.csv"
POI_TYPE_EMB_NPY = r"D:\ai 绿色\spatial-embedding\all_poi_embedding.npy"
POI_CSV = r"D:\ai 绿色\Data_for_Numpy\AMS_remove_zero\poi筛选后.csv"
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, f"checkpoint_no_revin_h{HORIZON}.pt")

# 数据加载
def load_all_data():
    temporal_df = pd.read_csv(PATH_TEMPORAL).select_dtypes(include=[np.number])
    weather_df = pd.read_csv(PATH_WEATHER).select_dtypes(include=[np.number])
    yday_df = pd.read_csv(PATH_YDAY).select_dtypes(include=[np.number])
    lweek_df = pd.read_csv(PATH_LWEEK).select_dtypes(include=[np.number])
    avg7_df = pd.read_csv(PATH_7AVG).select_dtypes(include=[np.number])
    targets_df = pd.read_csv(TARGET_FILE).select_dtypes(include=[np.number])

    temporal = torch.tensor(temporal_df.values, dtype=torch.float32)
    weather = torch.tensor(weather_df.values, dtype=torch.float32)
    lags = torch.stack([
        torch.tensor(yday_df.values, dtype=torch.float32),
        torch.tensor(lweek_df.values, dtype=torch.float32),
        torch.tensor(avg7_df.values, dtype=torch.float32)
    ], dim=-1)
    targets = torch.tensor(targets_df.values, dtype=torch.float32)
    return temporal, weather, lags, targets


_data_cache, _graph_cache = {}, {}


def load_data_once():
    global _data_cache, _graph_cache
    print("\n" + "=" * 60)
    print("正在预加载数据（仅一次）")
    print("=" * 60)

    if 'processed_data' not in _data_cache:
        print("正在读取数据文件...")
        temporal, weather, lags, targets = load_all_data()
        total_samples = temporal.shape[0] - SEQ_LEN - HORIZON + 1
        train_idx, val_idx, test_idx, purge_gap = make_purged_time_splits(
            total_samples, SEQ_LEN, HORIZON, TRAIN_RATIO, VAL_RATIO
        )

        print("使用训练集拟合 SiteAwareScaler，并标准化时间/天气协变量...")
        t_scaler, w_scaler = SiteAwareScaler(), SiteAwareScaler()
        train_input_end = train_idx[-1] + SEQ_LEN
        temporal_np = temporal.numpy()
        weather_np = weather.numpy()
        t_scaler.fit(temporal_np[:train_input_end])
        w_scaler.fit(weather_np[:train_input_end])
        temporal = torch.tensor(t_scaler.transform(temporal_np), dtype=torch.float32)
        weather = torch.tensor(w_scaler.transform(weather_np), dtype=torch.float32)
        print("使用训练集逐站点 z-score 标准化需求 lag 和目标值（无 RevIN）...")
        demand_scaler = DemandScaler()
        train_target_end = train_idx[-1] + SEQ_LEN + HORIZON
        targets_np = targets.numpy()
        lags_np = lags.numpy()
        demand_scaler.fit(targets_np[:train_target_end])
        lags = torch.tensor(
            np.stack([demand_scaler.transform(lags_np[..., i]) for i in range(lags_np.shape[-1])], axis=-1),
            dtype=torch.float32
        )
        targets = torch.tensor(demand_scaler.transform(targets_np), dtype=torch.float32)
        print(f"Split sizes after purge_gap={purge_gap}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        print("正在创建数据集...")
        dataset = EVChargingDataset(temporal, weather, lags, targets, SEQ_LEN, HORIZON)

        _data_cache['processed_data'] = {
            'dataset': dataset,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'purge_gap': purge_gap,
            'demand_scaler': demand_scaler,
        }

    if 'graph_data' not in _graph_cache:
        print("正在加载空间图...")
        graph = HeteroGraph(EDGE_CSV)
        try:
            all_poi_emb = np.load(POI_TYPE_EMB_NPY)
            poi_ids = pd.read_csv(POI_CSV)["ID"].astype(int).tolist()
            max_poi_id = max(poi_ids) if poi_ids else -1
            if all_poi_emb.shape[0] > max_poi_id:
                poi_emb = torch.tensor(all_poi_emb[poi_ids], dtype=torch.float32)
            elif all_poi_emb.shape[0] == len(poi_ids):
                poi_emb = torch.tensor(all_poi_emb, dtype=torch.float32)
            else:
                raise ValueError(
                    f"POI embedding rows ({all_poi_emb.shape[0]}) do not match POI ids up to {max_poi_id}."
                )
        except FileNotFoundError:
            print(f"Missing POI embedding file: {POI_TYPE_EMB_NPY}; using zeros.")
            poi_ids = pd.read_csv(POI_CSV)["ID"].astype(int).tolist()
            poi_emb = torch.zeros((len(poi_ids), 768), dtype=torch.float32)

        print("正在缓存空间图...")
        _graph_cache['graph_data'] = {
            'graph': graph, 'poi_emb': poi_emb,
            'site_ids': list(range(1, NUM_SITES + 1)),
            'poi_ids': poi_ids
        }
    print("数据预加载完成！\n")


class EVChargingDataset(Dataset):
    def __init__(self, temporal, weather, lags, targets, seq_len, horizon):
        self.temporal, self.weather, self.lags, self.targets = temporal, weather, lags, targets
        self.seq_len, self.horizon = seq_len, horizon

    def __len__(self): return self.temporal.shape[0] - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        return self.temporal[idx:idx + self.seq_len], self.weather[idx:idx + self.seq_len], \
            self.lags[idx:idx + self.seq_len], self.targets[idx + self.seq_len:idx + self.seq_len + self.horizon]

# 模型定义
class HeteroGraph:
    def __init__(self, edge_csv):
        df = pd.read_csv(edge_csv)
        self.node_list = sorted(set(df["src"]).union(set(df["dst"])))
        self.rel2id, self.edge_index = {}, defaultdict(list)
        for _, r in df.iterrows():
            s, d, rel = int(r["src"]), int(r["dst"]), r["edge_type"]
            if rel not in self.rel2id: self.rel2id[rel] = len(self.rel2id)
            self.edge_index[self.rel2id[rel]].append([s, d])
        for r in self.edge_index: self.edge_index[r] = torch.tensor(self.edge_index[r]).t().long()


class HGTLayer(nn.Module):
    def __init__(self, dim, num_rels, heads=4):
        super().__init__()
        self.dk, self.heads = dim // heads, heads
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
            score = (Q * K).sum(-1) / (self.dk ** 0.5) * self.rel_att[r]
            order = torch.argsort(dst)
            dst_sorted, score_sorted = dst[order], score[order]
            uniq, cnt = torch.unique_consecutive(dst_sorted, return_counts=True)
            alpha = torch.cat([torch.softmax(s, dim=0) for s in torch.split(score_sorted, cnt.tolist())], dim=0)
            msg = (alpha.unsqueeze(-1) * V[order]).reshape(-1, self.heads * self.dk)
            out.index_add_(0, dst_sorted, msg)
        return self.out(out)


class HGT(nn.Module):
    def __init__(self, graph, site_ids, poi_ids, poi_type_embedding, dim=64, num_layers=2, heads=4, device='cpu',
                 mlp_hidden_size=128, mlp_activation='relu'):
        super().__init__()
        self.device, self.graph = device, graph
        self.node2idx = {nid: i for i, nid in enumerate(sorted(graph.node_list))}
        self.num_sites, self.num_pois = len(site_ids), len(poi_ids)
        self.poi_ids_set = set(poi_ids)
        self.poi_id_to_idx = {int(nid): i for i, nid in enumerate(poi_ids)}
        self.node_emb = nn.Embedding(len(graph.node_list), dim).to(device)
        self.poi_features = nn.Parameter(poi_type_embedding.clone().detach().to(device), requires_grad=True)
        activation = nn.ReLU() if mlp_activation == 'relu' else nn.GELU()
        self.poi_feat_proj = nn.Sequential(nn.Linear(poi_type_embedding.shape[1], mlp_hidden_size), activation,
                                           nn.Linear(mlp_hidden_size, dim)).to(device)
        self.edge_index = {}
        for etype, idx in graph.edge_index.items():
            mapped = torch.zeros_like(idx, dtype=torch.long)
            for i in range(idx.shape[0]):
                for j in range(idx.shape[1]): mapped[i, j] = self.node2idx[idx[i, j].item()]
            self.edge_index[etype] = mapped.to(device)
        self.layers = nn.ModuleList([HGTLayer(dim, len(graph.rel2id), heads).to(device) for _ in range(num_layers)])
        self.linear = nn.Linear(dim, dim).to(device)
        self._cached_emb = None

    def forward(self):
        if self._cached_emb is None:
            x = self.node_emb.weight.clone()
            poi_emb = self.poi_feat_proj(self.poi_features)
            for i, nid in enumerate(self.graph.node_list):
                if nid in self.poi_ids_set:
                    poi_idx = self.poi_id_to_idx[int(nid)]
                    if 0 <= poi_idx < self.num_pois: x[i] = x[i] + poi_emb[poi_idx]
            for layer in self.layers: x = layer(x, self.edge_index)
            self._cached_emb = self.linear(x)
        return self._cached_emb

    def clear_cache(self):
        self._cached_emb = None


class TemporalTransformer(nn.Module):
    def __init__(self, time_dim, weather_dim, lag_dim, d_model=64, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.time_proj = nn.Sequential(nn.Linear(time_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
                                       nn.Dropout(dropout))
        self.weather_proj = nn.Sequential(nn.Linear(weather_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
                                          nn.Dropout(dropout))
        self.lag_proj = nn.Sequential(nn.Linear(lag_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
                                      nn.Dropout(dropout))
        self.gate_net = nn.Sequential(nn.Linear(3 * d_model, 3), nn.Sigmoid())
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal_feat, weather_feat, volume_lags):
        B, T, N, _ = volume_lags.shape
        t_emb, w_emb, l_emb = self.time_proj(temporal_feat), self.weather_proj(weather_feat), self.lag_proj(volume_lags)
        tw_emb = (t_emb + w_emb).unsqueeze(2).repeat(1, 1, N, 1)
        gates = self.gate_net(torch.cat([tw_emb, l_emb, tw_emb * l_emb], dim=-1)).softmax(dim=-1)
        fused = gates[..., 0:1] * tw_emb + gates[..., 1:2] * l_emb + gates[..., 2:3] * (tw_emb * l_emb)
        fused = self.dropout(fused).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        return self.transformer(fused).reshape(B, N, T, -1).permute(0, 2, 1, 3)


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(num_scales):
            kernel_size = 2 * (i + 1) + 1
            padding = kernel_size // 2
            self.convs.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.activations.append(nn.GELU())

        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, T, N, D = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * N, D, T)
        scale_features = []
        for i in range(self.num_scales):
            feat = self.convs[i](x_reshaped).permute(0, 2, 1)
            feat = self.norms[i](feat)
            feat = self.activations[i](feat)
            scale_features.append(feat)
        concatenated = torch.cat(scale_features, dim=-1)
        fused = self.layer_norm(self.fusion(concatenated))
        return fused.reshape(B, N, T, -1).permute(0, 2, 1, 3)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads, self.dk = heads, dim // heads
        self.Wq, self.Wk, self.Wv = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.fusion = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(dim * 2, dim))
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, time_emb, spatial_emb):
        B, T, N, D = time_emb.shape
        q = self.Wq(time_emb).view(B, T, N, self.heads, self.dk).permute(0, 3, 1, 2, 4)

        if spatial_emb.dim() == 4:
            k = self.Wk(spatial_emb).view(B, T, N, self.heads, self.dk).permute(0, 3, 1, 2, 4)
            v = self.Wv(spatial_emb).view(B, T, N, self.heads, self.dk).permute(0, 3, 1, 2, 4)
            spatial_ctx = spatial_emb
        else:
            k = self.Wk(spatial_emb).view(B, N, self.heads, self.dk).unsqueeze(2).repeat(1, 1, T, 1, 1).permute(0, 3, 2,
                                                                                                                1, 4)
            v = self.Wv(spatial_emb).view(B, N, self.heads, self.dk).unsqueeze(2).repeat(1, 1, T, 1, 1).permute(0, 3, 2,
                                                                                                                1, 4)
            spatial_ctx = spatial_emb.unsqueeze(1).repeat(1, T, 1, 1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.dk ** 0.5), dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 3, 1, 4).reshape(B, T, N, D)
        temporal_ctx = time_emb.mean(dim=1, keepdim=True).repeat(1, T, 1, 1)
        fused = self.fusion(torch.cat([out, spatial_ctx, temporal_ctx], dim=-1))
        return self.layer_norm(fused + time_emb)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class PatchTST(nn.Module):
    def __init__(self, in_d_model, d_model=64, patch=1, nhead=4, num_layers=2, dim_ff=256, horizon=1):
        super().__init__()
        self.patch, self.horizon, self.d_model = patch, horizon, d_model
        self.dim_proj = nn.Linear(in_d_model, d_model)
        self.patch_proj = nn.Linear(patch * d_model, d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True),
                                             num_layers)
        self.pool = AttentionPooling(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        B, T, N, _ = x.shape
        x = self.dim_proj(x)
        x = x.permute(0, 2, 1, 3).reshape(B * N, max(1, T // self.patch), self.patch * self.d_model)
        return self.head(self.pool(self.encoder(self.patch_proj(x)))).view(B, N, self.horizon).permute(0, 2, 1)


# ============================================================
# ============================================================
class EVChargingModel(nn.Module):
    def __init__(self, time_dim, weather_dim, lag_dim, spatial_encoder,
                 hgt_dim=32, temp_d_model=64, temp_nhead=4, patch_d_model=64, horizon=1, dropout=0.1,
                 temp_layers=1, fusion_heads=4, patch_nhead=4, patch_layers=2, patch_ff=256):
        super().__init__()

        self.temporal = TemporalTransformer(
            time_dim, weather_dim, lag_dim,
            d_model=temp_d_model, nhead=temp_nhead, num_layers=temp_layers, dropout=dropout
        )
        self.multi_scale = MultiScaleFeatureExtractor(input_dim=temp_d_model, hidden_dim=temp_d_model)
        self.spatial = spatial_encoder
        self.spatial_adapter = nn.Linear(hgt_dim, temp_d_model) if hgt_dim != temp_d_model else nn.Identity()
        self.time_modulation = nn.Linear(temp_d_model, temp_d_model)
        self.fusion = CrossAttentionFusion(temp_d_model, heads=fusion_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = PatchTST(
            temp_d_model, patch_d_model,
            nhead=patch_nhead, num_layers=patch_layers, dim_ff=patch_ff, horizon=horizon
        )

    def forward(self, temporal_x, weather_x, lags_x):
        time_emb = self.temporal(temporal_x, weather_x, lags_x)
        multi_scale_emb = self.multi_scale(time_emb)
        B, T, N, D = multi_scale_emb.shape

        site_emb = self.spatial_adapter(self.spatial()[:N]).unsqueeze(0).repeat(B, 1, 1)
        global_temporal = multi_scale_emb.mean(dim=2)
        modulated_time = self.time_modulation(global_temporal).unsqueeze(2).repeat(1, 1, N, 1)

        dynamic_site_emb = site_emb.unsqueeze(1).repeat(1, T, 1, 1) + modulated_time
        fused = self.dropout(self.fusion(multi_scale_emb, dynamic_site_emb))

        return self.head(fused)


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience, self.best, self.counter, self.early_stop = patience, None, 0, False

    def __call__(self, metric):
        if self.best is None or metric < self.best:
            self.best, self.counter = metric, 0
        else:
            self.counter += 1
        if self.counter >= self.patience: self.early_stop = True
        return self.early_stop

# Optuna 超参数搜索
def objective(trial):
    global DEVICE, _data_cache, _graph_cache
    cached_data, cached_graph = _data_cache['processed_data'], _graph_cache['graph_data']
    dataset, graph, poi_emb, site_ids, poi_ids = cached_data['dataset'], cached_graph['graph'], cached_graph['poi_emb'], \
        cached_graph['site_ids'], cached_graph['poi_ids']

    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    hgt_dim = trial.suggest_categorical('hgt_dim', [16, 32])
    temp_d_model = trial.suggest_categorical('temp_d_model', [64, 128])
    temp_nhead = trial.suggest_categorical('temp_nhead', [2, 4])
    temp_layers = trial.suggest_categorical('temp_layers', [1, 2])
    fusion_heads = trial.suggest_categorical('fusion_heads', [2, 4])
    patch_d_model = trial.suggest_categorical('patch_d_model', [32, 64])
    patch_nhead = trial.suggest_categorical('patch_nhead', [2, 4])
    patch_layers = trial.suggest_categorical('patch_layers', [1, 2])
    patch_ff = trial.suggest_categorical('patch_ff', [128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    mlp_hidden_size = trial.suggest_categorical('mlp_hidden_size', [64, 128])
    mlp_activation = trial.suggest_categorical('mlp_activation', ['relu', 'gelu'])

    if temp_d_model % temp_nhead != 0:
        raise optuna.TrialPruned()
    if temp_d_model % fusion_heads != 0:
        raise optuna.TrialPruned()
    if patch_d_model % patch_nhead != 0:
        raise optuna.TrialPruned()

    train_loader = DataLoader(Subset(dataset, cached_data['train_idx']), BATCH_SIZE, shuffle=True,
                              pin_memory=PIN_MEMORY)
    val_loader = DataLoader(Subset(dataset, cached_data['val_idx']), BATCH_SIZE, pin_memory=PIN_MEMORY)

    try:
        hgt_model = HGT(
            graph, site_ids, poi_ids, poi_emb, dim=hgt_dim, device=DEVICE,
            mlp_hidden_size=mlp_hidden_size, mlp_activation=mlp_activation
        ).to(DEVICE)
        model = EVChargingModel(dataset.temporal.shape[-1], dataset.weather.shape[-1], dataset.lags.shape[-1],
                                hgt_model, hgt_dim, temp_d_model, temp_nhead, patch_d_model, HORIZON, dropout_rate,
                                temp_layers, fusion_heads, patch_nhead, patch_layers, patch_ff).to(
            DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn, early_stopper = nn.MSELoss(), EarlyStopping(patience=4)
        best_val_loss = float("inf")

        for epoch in range(OPTUNA_EPOCHS):
            model.train()
            for x1, x2, x3, y in train_loader:
                optimizer.zero_grad()
                pred = model(x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE))
                y_device = y.to(DEVICE)
                assert pred.shape == y_device.shape, (pred.shape, y_device.shape)
                loss = loss_fn(pred, y_device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.spatial.clear_cache()

            model.eval()
            val_losses = []
            model.spatial.clear_cache()
            with torch.no_grad():
                for x1, x2, x3, y in val_loader:
                    pred = model(x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE))
                    y_device = y.to(DEVICE)
                    assert pred.shape == y_device.shape, (pred.shape, y_device.shape)
                    val_losses.append(loss_fn(pred, y_device).item())
            model.spatial.clear_cache()

            val_loss = np.mean(val_losses)
            best_val_loss = min(best_val_loss, val_loss)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if early_stopper(val_loss): break

        return best_val_loss
    finally:
        if 'model' in locals(): del model, hgt_model
        torch.cuda.empty_cache()

# 主程序
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"运行无 RevIN 对照实验，结果目录：{RESULTS_DIR}")

    load_data_once()

    if optuna is not None and RUN_OPTUNA:
        print(f"\n开始为 HORIZON={HORIZON} 进行超参数搜索...")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=3, interval_steps=1),
        )
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        best_params = study.best_params
        print(f"\n超参数搜索完成，最佳参数：{best_params}")
    else:
        print("使用默认超参数...")
        best_params = {
            'lr': 1e-3,
            'weight_decay': 1e-2,
            'hgt_dim': 32,
            'temp_d_model': 64,
            'temp_nhead': 4,
            'temp_layers': 1,
            'fusion_heads': 4,
            'patch_d_model': 32,
            'patch_nhead': 4,
            'patch_layers': 2,
            'patch_ff': 256,
            'dropout_rate': 0.3,
            'mlp_hidden_size': 128,
            'mlp_activation': 'relu',
        }

    # 3. 构建数据加载器
    cached_data, cached_graph = _data_cache['processed_data'], _graph_cache['graph_data']
    dataset = cached_data['dataset']
    demand_scaler = cached_data['demand_scaler']

    train_loader = DataLoader(Subset(dataset, cached_data['train_idx']), BATCH_SIZE, shuffle=True,
                              pin_memory=PIN_MEMORY)
    val_loader = DataLoader(Subset(dataset, cached_data['val_idx']), BATCH_SIZE, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(Subset(dataset, cached_data['test_idx']), BATCH_SIZE, pin_memory=PIN_MEMORY)

    # 4. 初始化模型
    print("正在初始化模型...")
    hgt_model = HGT(cached_graph['graph'], cached_graph['site_ids'], cached_graph['poi_ids'], cached_graph['poi_emb'],
                    dim=best_params['hgt_dim'], device=DEVICE,
                    mlp_hidden_size=best_params['mlp_hidden_size'],
                    mlp_activation=best_params['mlp_activation']).to(DEVICE)
    model = EVChargingModel(dataset.temporal.shape[-1], dataset.weather.shape[-1], dataset.lags.shape[-1],
                            hgt_model, best_params['hgt_dim'], best_params['temp_d_model'], best_params['temp_nhead'],
                            best_params['patch_d_model'], HORIZON, best_params['dropout_rate'],
                            best_params['temp_layers'], best_params['fusion_heads'], best_params['patch_nhead'],
                            best_params['patch_layers'], best_params['patch_ff']).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for x1, x2, x3, y in train_loader:
            optimizer.zero_grad()
            pred = model(x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE))
            y_device = y.to(DEVICE)
            assert pred.shape == y_device.shape, (pred.shape, y_device.shape)
            loss = loss_fn(pred, y_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.spatial.clear_cache()
            train_losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(train_losses)

        if epoch % 2 == 0:
            model.eval()
            val_losses = []
            val_trues, val_preds = [], []
            with torch.no_grad():
                for x1, x2, x3, y in val_loader:
                    pred = model(x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE))
                    y_device = y.to(DEVICE)
                    assert pred.shape == y_device.shape, (pred.shape, y_device.shape)
                    val_losses.append(loss_fn(pred, y_device).item())
                    val_trues.append(y.cpu().numpy())
                    val_preds.append(pred.cpu().numpy())
                    model.spatial.clear_cache()

            val_loss = np.mean(val_losses)

            val_trues = np.concatenate(val_trues, 0)
            val_preds = np.concatenate(val_preds, 0)
            val_trues = demand_scaler.inverse_transform(val_trues)
            val_preds = demand_scaler.inverse_transform(val_preds)
            val_global_r2, val_mean_r2, val_median_r2, val_rmse, val_mae = compute_metrics(val_trues, val_preds)

            print(f"Epoch {epoch + 1:02d} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f} | "
                  f"Avg Site R2 {val_mean_r2:.4f} | Median Site R2 {val_median_r2:.4f} | "
                  f"RMSE {val_rmse:.4f} | MAE {val_mae:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state": model.state_dict()}, CHECKPOINT_FILE)
            if early_stopper(val_loss):
                print("触发早停。")
                break

    model.load_state_dict(torch.load(CHECKPOINT_FILE)["model_state"])
    model.eval()
    test_losses, test_trues, test_preds = [], [], []
    with torch.no_grad():
        for x1, x2, x3, y in test_loader:
            y = y.to(DEVICE)
            pred = model(x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE))
            assert pred.shape == y.shape, (pred.shape, y.shape)
            test_losses.append(loss_fn(pred, y).item())
            test_trues.append(y.cpu().numpy())
            test_preds.append(pred.cpu().numpy())
            model.spatial.clear_cache()

    test_trues_scaled = np.concatenate(test_trues, 0).reshape(-1, HORIZON, NUM_SITES)
    test_preds_scaled = np.concatenate(test_preds, 0).reshape(-1, HORIZON, NUM_SITES)
    test_trues_inv = demand_scaler.inverse_transform(test_trues_scaled)
    test_preds_inv = demand_scaler.inverse_transform(test_preds_scaled)

    metrics_test = compute_metrics(test_trues_inv, test_preds_inv)
    horizon_metrics = compute_horizon_metrics(test_trues_inv, test_preds_inv)

    print("\n" + "=" * 40)
    print("最终测试结果")
    print("=" * 40)
    print(f"Test Loss {np.mean(test_losses):.6f}")
    print(f"Global R2  : {metrics_test[0]:.4f} (affected by cross-site scale differences)")
    print(f"Avg Site R2: {metrics_test[1]:.4f} (primary station-wise metric)")
    print(f"Med Site R2: {metrics_test[2]:.4f}")
    print(f"Test RMSE  : {metrics_test[3]:.4f}")
    print(f"Test MAE   : {metrics_test[4]:.4f}")
    print(f"Horizon-wise: {format_horizon_metrics(horizon_metrics)}")

    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Test Metrics (Horizon={HORIZON}, without RevIN)\n")
        f.write("Demand normalization: train-only per-site z-score; reported metrics are inverse-transformed.\n")
        f.write(f"Split sizes: train={len(cached_data['train_idx'])}, val={len(cached_data['val_idx'])}, test={len(cached_data['test_idx'])}\n")
        f.write(f"Purge gap: {cached_data['purge_gap']}\n")
        f.write(f"Loss: {np.mean(test_losses):.6f}\n")
        f.write(f"Global R2: {metrics_test[0]:.4f}\n")
        f.write(f"Avg Site R2: {metrics_test[1]:.4f}\n")
        f.write(f"Median Site R2: {metrics_test[2]:.4f}\n")
        f.write(f"RMSE: {metrics_test[3]:.4f}\n")
        f.write(f"MAE: {metrics_test[4]:.4f}\n\n")

        f.write("Horizon-wise Metrics\n")
        for m in horizon_metrics:
            f.write(
                f"H{m['horizon']}: Global R2={m['global_r2']:.4f}, "
                f"Avg Site R2={m['avg_site_r2']:.4f}, Median Site R2={m['median_site_r2']:.4f}, "
                f"RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}\n"
            )
        f.write("\n")

        f.write("Best Hyperparameters\n")
        if optuna is not None:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")

    print(f"测试指标和最佳参数已保存至：{metrics_path}")

    print("\n正在打包测试集预测数据，供画图脚本使用...")
    edges_np = {str(k): v.cpu().numpy() for k, v in cached_graph['graph'].edge_index.items()}
    npz_save_path = os.path.join(results_dir, "PlotData.npz")

    flat_trues = test_trues_inv[:, 0, :]  # 取第一个预测步，形状为 [TimeSteps, NumSites]
    flat_preds = test_preds_inv[:, 0, :]  # 取第一个预测步，形状为 [TimeSteps, NumSites]

    np.savez(
        npz_save_path,
        trues=flat_trues,
        preds=flat_preds,
        num_sites=NUM_SITES,
        horizon=HORIZON,
        **edges_np
    )
    print(f"画图所需数据已保存至：{npz_save_path}")
