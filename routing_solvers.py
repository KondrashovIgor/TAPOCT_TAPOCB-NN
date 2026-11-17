# -*- coding: utf-8 -*-
"""
routing_solvers.py — солверы портфеля для TAPOCT-FD (собрано из ячеек 5, 6, 7 ноутбука
TAPOCT_TAPOCB-NN_main.ipynb без изменения логики) + фабрика get_solvers().

Зачем отдельный модуль: joblib/loky сериализует функции из __main__ ноутбука ПО ЗНАЧЕНИЮ вместе
со всеми глобалами, на которые они ссылаются (MATNET_MODEL, NLNS_MODEL_LKH, distance_array, ...).
Функции из импортируемого модуля сериализуются ПО ССЫЛКЕ: каждый воркер импортирует модуль
(и загружает веса на GPU) ровно один раз.

ВНИМАНИЕ: пути PATH_MATNET, NLNS_PATH_LKH, lkh_dir взяты из ноутбука — проверьте их.

Схема:
get_solvers() -> {"MatNet": _matnet, "NLNS": _nlns, "lkh3": _lkh3, "Christofides": _christofides}
  _matnet        -> solve_MatNet_atsp -> run_matnet_inference -> MatNet.decode / tour_cost
  _nlns          -> solve_NLNS_lkh_atsp -> run_nlns_pipeline_lkh -> solve_lkh_atsp (warm start) -> NLNS_Model.solve
  _lkh3          -> solve_lkh_atsp(runs=5)
  _christofides  -> solve_TSP_Christofides_inner (OR-Tools, CHRISTOFIDES first solution)
Все обёртки: (points: list[id] длины n+1, prev_point: id, cost_matrix: (n+1,n+1)) -> (route: list[id], cost: float)
"""
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# ======================= ячейка 5 ноутбука =======================
import math
import copy
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Mixed-Score Multi-Head Attention  (THE core of MatNet)
#    Fuses the scaled dot-product score with the raw matrix
#    entry (distance) through a tiny MLP "mixer".
# ============================================================
class MixedScoreMHA(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, ms_hidden: int = 16):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, embed_dim // n_heads
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        # mixer: [dot_score, matrix_entry] -> fused score  (per head)
        self.mixer = nn.Sequential(nn.Linear(2, ms_hidden), nn.ReLU(),
                                   nn.Linear(ms_hidden, 1))

    def forward(self, q_in, kv_in, matrix, key_mask=None):
        # q_in:(B,Nq,E)  kv_in:(B,Nk,E)  matrix:(B,Nq,Nk)  key_mask:(B,Nk) True=valid
        B, Nq, _ = q_in.shape
        Nk = kv_in.size(1)
        H, hd = self.n_heads, self.head_dim
        q = self.Wq(q_in).view(B, Nq, H, hd).transpose(1, 2)
        k = self.Wk(kv_in).view(B, Nk, H, hd).transpose(1, 2)
        v = self.Wv(kv_in).view(B, Nk, H, hd).transpose(1, 2)
        dot = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)   # (B,H,Nq,Nk)
        mat = matrix.unsqueeze(1).expand(-1, H, -1, -1)              # (B,H,Nq,Nk)
        scores = self.mixer(torch.stack([dot, mat], dim=-1)).squeeze(-1)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Nq, -1)
        return self.out(out)


# ============================================================
# 2. Dual-attention encoder block (rows <-> columns)
# ============================================================
class MatNetLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, ms_hidden):
        super().__init__()
        self.row_attn = MixedScoreMHA(embed_dim, n_heads, ms_hidden)
        self.col_attn = MixedScoreMHA(embed_dim, n_heads, ms_hidden)
        self.rn1, self.rn2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.cn1, self.cn2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.rff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(),
                                 nn.Linear(ff_dim, embed_dim))
        self.cff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(),
                                 nn.Linear(ff_dim, embed_dim))

    def forward(self, row, col, matrix, mask):
        # rows attend to columns through D ; columns attend to rows through D^T
        row = self.rn1(row + self.row_attn(row, col, matrix, mask))
        row = self.rn2(row + self.rff(row))
        col = self.cn1(col + self.col_attn(col, row, matrix.transpose(1, 2), mask))
        col = self.cn2(col + self.cff(col))
        return row, col


# ============================================================
# 3. Autoregressive decoder (Attention-Model style:
#    multi-head glimpse + single-head clipped logits)
# ============================================================
class MatNetDecoder(nn.Module):
    def __init__(self, embed_dim, n_heads, clip=10.0):
        super().__init__()
        self.n_heads, self.head_dim, self.clip = n_heads, embed_dim // n_heads, clip
        self.Wq   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.WkL  = nn.Linear(embed_dim, embed_dim, bias=False)

    def set_kv(self, node_emb):                       # cache K,V once per instance
        B, N, E = node_emb.shape
        H, hd = self.n_heads, self.head_dim
        self.gk = self.Wk(node_emb).view(B, N, H, hd).transpose(1, 2)
        self.gv = self.Wv(node_emb).view(B, N, H, hd).transpose(1, 2)
        self.lk = self.WkL(node_emb)
        self.scale = math.sqrt(E)

    def forward(self, query, visited):                # query:(B,E) visited:(B,N) bool
        B, E = query.shape
        H, hd = self.n_heads, self.head_dim
        q = self.Wq(query).view(B, H, 1, hd)
        s = torch.matmul(q, self.gk.transpose(-2, -1)) / math.sqrt(hd)
        s = s.masked_fill(visited[:, None, None, :], -1e9)
        glimpse = torch.matmul(F.softmax(s, dim=-1), self.gv)         # (B,H,1,hd)
        glimpse = self.Wout(glimpse.transpose(1, 2).reshape(B, 1, E)).squeeze(1)
        logits = torch.matmul(glimpse.unsqueeze(1),
                              self.lk.transpose(1, 2)).squeeze(1) / self.scale
        logits = self.clip * torch.tanh(logits)
        return logits.masked_fill(visited, -1e9)


# ============================================================
# 4. The full MatNet model
# ============================================================
class MatNet(nn.Module):
    def __init__(self, embed_dim=128, n_layers=5, n_heads=8,
                 ff_dim=512, ms_hidden=16, clip=10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList(
            [MatNetLayer(embed_dim, n_heads, ff_dim, ms_hidden) for _ in range(n_layers)])
        self.decoder = MatNetDecoder(embed_dim, n_heads, clip)

    # MatNet initial embeddings: rows = zeros, columns = random one-hot ("instance
    # augmentation" replacement of node features for a featureless matrix).
    def _init_emb(self, B, N, device):
        """Начальные эмбеддинги узлов, включая N > embed_dim."""
        row = torch.zeros(B, N, self.embed_dim, device=device)
        col = torch.zeros(B, N, self.embed_dim, device=device)
    
        base = torch.rand(B, self.embed_dim, device=device).argsort(dim=1)
        if N <= self.embed_dim:
            idx = base[:, :N]
        else:
            # Для дополнительных узлов повторяются индексы базиса.
            extra = torch.randint(
                self.embed_dim, (B, N - self.embed_dim), device=device
            )
            idx = torch.cat([base, extra], dim=1)
    
        col.scatter_(2, idx.unsqueeze(-1), 1.0)
        return row, col

    def encode(self, dist, mask):
        B, N, _ = dist.shape
        # 1. Mask out the padded nodes BEFORE finding the maximum
        # mask shape is (B, N), we need it to be (B, 1, 1) or broadcastable to (B, N, N)
        valid_edge_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        masked_dist = dist.masked_fill(~valid_edge_mask, 0.0)
        
        # 2. Calculate max ONLY on real nodes
        max_val = masked_dist.amax(dim=(1, 2), keepdim=True) + 1e-8
        
        # 3. Normalize
        mnorm = dist / max_val 
        
        row, col = self._init_emb(B, N, dist.device)
        for layer in self.layers:
            row, col = layer(row, col, mnorm, mask)
        return row, col

    def decode(self, dist, mask, n_nodes, start_idx=None,
               decode_type='greedy', temperature=1.0):
        """Returns (tours:(B,N_max), sum_log_prob:(B,)). OPEN tour: no return edge."""
        B, N, _ = dist.shape
        device = dist.device
        if start_idx is None:
            start_idx = torch.zeros(B, dtype=torch.long, device=device)
        row, _ = self.encode(dist, mask)
        self.decoder.set_kv(row)

        # NOTE: every update below is OUT-OF-PLACE so that the `visited`/`logits`
        # tensors saved by masked_fill for backward are never mutated afterwards.
        visited = (~mask).clone()                              # (B,N) bool
        visited = visited.scatter(1, start_idx[:, None], True) # out-of-place
        ar = torch.arange(B, device=device)
        cur = start_idx
        tours, logps = [start_idx], []

        for _ in range(N - 1):
            logits = self.decoder(row[ar, cur], visited)       # (B,N), grad-tracked
            done = (logits <= -1e8).all(-1)                    # finished instances
            if done.any():
                # out-of-place self-loop fix (logp = 0 for finished rows)
                self_mask = torch.zeros_like(visited)
                self_mask[ar[done], cur[done]] = True          # fresh bool tensor, no grad
                logits = torch.where(self_mask,
                                     torch.zeros_like(logits), logits)

            probs = F.softmax(logits / temperature, -1)
            if decode_type == 'greedy':
                nxt = probs.argmax(-1)
            else:
                nxt = torch.distributions.Categorical(probs).sample()

            logps.append(torch.log(probs.gather(1, nxt[:, None]).squeeze(1) + 1e-12))
            tours.append(nxt)
            visited = visited.scatter(1, nxt[:, None], True)   # out-of-place -> NEW tensor
            cur = nxt

        return torch.stack(tours, 1), torch.stack(logps, 1).sum(1)


# ============================================================
# 5. OPEN-ATSP tour cost  (path, NOT a cycle -> no last->first edge)
# ============================================================
def tour_cost(tours: torch.Tensor, dist: torch.Tensor, n_nodes: List[int]) -> torch.Tensor:
    B = tours.size(0)
    costs = dist.new_zeros(B)
    for b in range(B):
        n = n_nodes[b]
        seq = tours[b, :n]
        costs[b] = dist[b, seq[:-1], seq[1:]].sum()
    return costs


# ============================================================
# 6. Dataset + collate (pad ragged ATSP matrices to max_nodes)
# ============================================================
class ATSPDataset(Dataset):
    def __init__(self, polygon_ids, AUG_global_map, optimal_routes=None):
        self.ids = list(polygon_ids)
        self.maps = AUG_global_map
        self.opt = optimal_routes or {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        d = np.asarray(self.maps[pid]['distance matrix'], dtype=np.float32)
        return {'polygon_id': pid,
                'dist_matrix': torch.from_numpy(d),
                'n_nodes': d.shape[0]}


def make_collate(max_nodes=120):
    def collate(batch):
        B = len(batch)
        dist = torch.full((B, max_nodes, max_nodes), 1e4)
        mask = torch.zeros(B, max_nodes, dtype=torch.bool)
        n_nodes, ids = [], []
        for i, it in enumerate(batch):
            n = it['n_nodes']
            dist[i, :n, :n] = it['dist_matrix']
            mask[i, :n] = True
            n_nodes.append(n)
            ids.append(it['polygon_id'])
        return {'dist_matrices': dist, 'masks': mask,
                'n_nodes': n_nodes, 'polygon_ids': ids}
    return collate

device = 'cuda'
PATH_MATNET = r'C:\Users\Administrator\matnet_open_atsp_shuffled_fixed.pth'

MATNET_MODEL = None


def _load_matnet():
    """Загрузить MatNet один раз при первом вызове в данном воркере."""
    global MATNET_MODEL

    if MATNET_MODEL is None:
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MatNet(embed_dim=128, n_layers=5, n_heads=8, ff_dim=512)
        state = torch.load(PATH_MATNET, map_location=model_device, weights_only=True)
        model.load_state_dict(state)
        MATNET_MODEL = model.to(model_device).eval()

    return MATNET_MODEL, next(MATNET_MODEL.parameters()).device

print(f"MatNet initialized and moved to {device}")

def run_matnet_inference(dist_matrix: np.ndarray, model: MatNet,
                         device='cuda', start_idx=0,
                         decode_type='greedy', n_samples=1
                         ) -> Tuple[np.ndarray, float]:
    model.eval()
    N = dist_matrix.shape[0]
    dist = torch.as_tensor(dist_matrix, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)
    start = torch.tensor([start_idx], dtype=torch.long, device=device)

    if decode_type == 'greedy':
        t, _ = model.decode(dist, mask, [N], start, 'greedy')
        tour = t.squeeze(0).cpu().numpy()
        cost = float(tour_cost(t, dist, [N]).item())
    else:  # sampling: keep best of n_samples
        best_tour, best_cost = None, float('inf')
        for _ in range(n_samples):
            t, _ = model.decode(dist.expand(1, N, N), mask, [N], start, 'sampling')
            c = float(tour_cost(t, dist, [N]).item())
            if c < best_cost:
                best_cost, best_tour = c, t.squeeze(0).cpu().numpy()
        tour, cost = best_tour, best_cost
    return tour, cost

# ======================= ячейка 6 ноутбука =======================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
import re
import math
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict

# ==========================================
# ENCODER (Fixed NumPy sqrt leak)
# ==========================================
class NLNS_EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 128, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, dist_matrix=None, mask=None):
        B, N, _ = x.shape
        residual = x
        Q = self.Wq(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, -1)
        out = self.Wout(out)
        
        x = self.norm1(residual + out)
        x = self.norm2(x + self.ff(x))
        return x

class NLNS_Encoder(nn.Module):
    def __init__(self, embed_dim=128, n_layers=3, n_heads=8):
        super().__init__()
        self.feat_proj = nn.Linear(2, embed_dim)
        self.layers = nn.ModuleList([NLNS_EncoderLayer(embed_dim, n_heads) for _ in range(n_layers)])
        
    def forward(self, dist_matrix):
        B, N, _ = dist_matrix.shape
        row_mean = dist_matrix.mean(dim=-1, keepdim=True)
        row_std = dist_matrix.std(dim=-1, keepdim=True)
        feats = torch.cat([row_mean, row_std], dim=-1)
        
        x = self.feat_proj(feats)
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
# DECODER (Ensured strict device inheritance)
# ==========================================
class NLNS_RepairDecoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        E = embed_dim
        self.edge_proj = nn.Linear(2 * E, E, bias=False)
        self.q_proj = nn.Linear(E, E, bias=False)
        self.k_proj = nn.Linear(E, E, bias=False)
        self.cost_proj = nn.Linear(1, E, bias=False)
        self.scale = E ** 0.5

    def forward(self, v_idx, v_emb, partial_tour_embs, partial_tour_indices, dist_matrix):
        B, L, E = partial_tour_embs.shape
        device = dist_matrix.device
        
        next_indices = torch.roll(partial_tour_indices, shifts=-1, dims=1)
        head_e = partial_tour_embs
        tail_e = torch.roll(partial_tour_embs, shifts=-1, dims=1)
        
        edge_e = self.edge_proj(torch.cat([head_e, tail_e], dim=-1)) # (B, L, E)
        q = self.q_proj(v_emb) # (B, 1, E)
        k = self.k_proj(edge_e) # (B, L, E)
        
        attn_scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1) / self.scale # (B, L)
        
        bidx = torch.arange(B, device=device).view(B, 1)
        v_idx_exp = v_idx.view(B, 1).expand(B, L)
        
        cost_pv = dist_matrix[bidx, partial_tour_indices, v_idx_exp]
        cost_vp = dist_matrix[bidx, v_idx_exp, next_indices]
        cost_pp = dist_matrix[bidx, partial_tour_indices, next_indices]
        
        cost_change = cost_pv + cost_vp - cost_pp # (B, L)
        cost_emb = self.cost_proj(cost_change.unsqueeze(-1)) # (B, L, E)
        cost_score = (q * cost_emb).sum(-1) / self.scale # (B, L)
        
        scores = attn_scores - cost_score
        return scores

# ==========================================================
# FULL NLNS MODEL (destroy + neural repair)
# ==========================================================
class NLNS_Model(nn.Module):
    def __init__(self, embed_dim=128, n_layers=3, n_heads=8):
        super().__init__()
        self.encoder = NLNS_Encoder(embed_dim, n_layers, n_heads)
        self.decoder = NLNS_RepairDecoder(embed_dim)
        self.embed_dim = embed_dim

    @staticmethod
    def destroy(tours, destroy_fraction=0.2, start_idx=0):
        B, N = tours.shape
        k = max(1, int(N * destroy_fraction))
        
        removed_nodes_batch = []
        partial_tours_batch = []
        
        for b in range(B):
            tour_b = tours[b].tolist()
            candidates = [node for node in tour_b if node != start_idx]
            removed = random.sample(candidates, min(k, len(candidates)))
            removed_nodes_batch.append(removed)
            
            partial = [node for node in tour_b if node not in removed]
            partial_tours_batch.append(partial)
            
        return removed_nodes_batch, partial_tours_batch

    def repair(self, dist_matrix, tours, destroy_fraction=0.2, start_idx=0, greedy=False, temperature=1.0):
        B, N = tours.shape
        device = dist_matrix.device
        
        node_embs = self.encoder(dist_matrix)
        removed_nodes_batch, partial_tours_batch = self.destroy(tours, destroy_fraction, start_idx)
        
        batch_logp = []
        batch_ent = []
        final_tours = []
        costs = []
        
        for b in range(B):
            partial_tour = partial_tours_batch[b]
            removed_nodes = removed_nodes_batch[b]
            random.shuffle(removed_nodes)
            
            traj_logps = []
            traj_ents = []
            
            for v in removed_nodes:
                v_idx = torch.tensor([v], dtype=torch.long, device=device)
                v_emb = node_embs[b, v].unsqueeze(0).unsqueeze(0) # (1, 1, E)
                
                partial_tour_indices = torch.tensor(partial_tour, dtype=torch.long, device=device).unsqueeze(0) # (1, L)
                partial_tour_embs = node_embs[b, partial_tour_indices.squeeze(0)].unsqueeze(0) # (1, L, E)
                
                dist_b = dist_matrix[b].unsqueeze(0) # (1, N, N)
                
                scores = self.decoder(v_idx, v_emb, partial_tour_embs, partial_tour_indices, dist_b).squeeze(0) # (L,)
                
                if greedy:
                    pos_idx = torch.argmax(scores).item()
                    log_prob = torch.tensor(0.0, device=device)
                    ent = torch.tensor(0.0, device=device)
                else:
                    probs = F.softmax(scores / temperature, dim=-1)
                    probs = torch.clamp(probs, min=1e-8, max=1.0)
                    dist = torch.distributions.Categorical(probs)
                    
                    pos_idx_tensor = dist.sample()
                    pos_idx = pos_idx_tensor.item()
                    log_prob = dist.log_prob(pos_idx_tensor)
                    ent = dist.entropy()
                    
                traj_logps.append(log_prob)
                traj_ents.append(ent)
                
                partial_tour.insert(pos_idx + 1, v)
                
            final_tours.append(partial_tour)
            batch_logp.append(torch.stack(traj_logps).sum() if traj_logps else torch.tensor(0.0, device=device))
            batch_ent.append(torch.stack(traj_ents).sum() if traj_ents else torch.tensor(0.0, device=device))
            
            cost = 0.0
            for i in range(len(partial_tour) - 1):
                cost += dist_matrix[b, partial_tour[i], partial_tour[i+1]].item()
            costs.append(cost)
            
        final_tours_tensor = torch.tensor(final_tours, dtype=torch.long, device=device)
        costs_tensor = torch.tensor(costs, dtype=torch.float, device=device)
        total_logp = torch.stack(batch_logp) # Shape (B,)
        total_ent = torch.stack(batch_ent)   # Shape (B,)
            
        return final_tours_tensor, costs_tensor, total_logp, total_ent

    @torch.no_grad()
    def solve(self, dist_matrix, init_tours, destroy_fraction=0.2, n_iterations=20,
              start_idx=0, sample=True, temperature=1.0):
        self.eval()
        B, N = init_tours.shape
        device = dist_matrix.device
        best_tours = init_tours.clone()
        idx = torch.arange(B, device=device)
        
        best_cost = torch.zeros(B, device=device)
        for i in range(N - 1):
            best_cost += dist_matrix[idx, best_tours[:, i], best_tours[:, i + 1]]

        cur_tours, cur_cost = best_tours.clone(), best_cost.clone()
        
        for _ in range(n_iterations):
            new_tours, new_cost, _, _ = self.repair(
                dist_matrix, cur_tours, destroy_fraction=destroy_fraction,
                start_idx=start_idx, greedy=not sample, temperature=temperature)
            
            improved = new_cost < cur_cost
            cur_tours = torch.where(improved.unsqueeze(1), new_tours, cur_tours)
            cur_cost  = torch.where(improved, new_cost, cur_cost)
            
            better = cur_cost < best_cost
            best_tours = torch.where(better.unsqueeze(1), cur_tours, best_tours)
            best_cost  = torch.where(better, cur_cost, best_cost)
            
        return best_tours, best_cost


# ==========================================================
# INITIALIZATION & INFERENCE PIPELINE (100% CUDA Safe)
# ==========================================================
device = 'cuda'
NLNS_PATH_LKH = r'C:\Users\Administrator\nlns_open_atsp_lkh.pth'

NLNS_MODEL_LKH = None


def _load_nlns():
    """Загрузить NLNS один раз при первом вызове в данном воркере."""
    global NLNS_MODEL_LKH

    if NLNS_MODEL_LKH is None:
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NLNS_Model(embed_dim=128, n_layers=3, n_heads=8)
        state = torch.load(NLNS_PATH_LKH, map_location=model_device, weights_only=True)
        model.load_state_dict(state)
        NLNS_MODEL_LKH = model.to(model_device).eval()

    return NLNS_MODEL_LKH, next(NLNS_MODEL_LKH.parameters()).device
print(f"NLNS initialized and locked to {device}")

def run_nlns_pipeline_lkh(cost_matrix, orders, model,
                      start_idx=0, destroy_fraction=0.2, n_iterations=50, device='cuda'):
    model.eval()
    
    # Ensure cost_matrix is a numpy array before passing to heuristic
    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix_np = cost_matrix.cpu().numpy()
    else:
        cost_matrix_np = cost_matrix
        
    warm, warm_cost = solve_lkh_atsp(orders, orders[0], cost_matrix_np, runs=2)
    
    id2idx = {oid: i for i, oid in enumerate(orders)}
    idxd = [id2idx[o] for o in warm]
    
    dist = torch.tensor(cost_matrix_np, dtype=torch.float32, device=device).unsqueeze(0)
    warm_t = torch.tensor(idxd, dtype=torch.long, device=device).unsqueeze(0)

    best_tours, best_cost = model.solve(
        dist, warm_t, destroy_fraction=destroy_fraction,
        n_iterations=n_iterations, start_idx=start_idx, sample=True)

    return (best_tours.squeeze(0).cpu().numpy(),
            float(best_cost.squeeze(0).cpu()),
            float(warm_cost))

# ======================= ячейка 7 ноутбука =======================
#Alternative!
from ortools.linear_solver import pywraplp
import uuid
import tempfile
import os
import subprocess
import numpy as np
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

def solve_MatNet_atsp(points, prev_point, matrix_to_solve, model = MATNET_MODEL, device = 'cuda'):
    dist_matrix = matrix_to_solve
    result_route_idxs, route_distance = run_matnet_inference(dist_matrix, model, device = 'cuda')
    result_route = [points[i] for i in result_route_idxs]
    start_point = result_route[0]
    end_point = result_route[-1]
    
    return result_route, route_distance, start_point, end_point

def solve_NLNS_lkh_atsp(points, prev_point, matrix_to_solve, model = NLNS_MODEL_LKH, device = 'cuda'):
    if prev_point in points:
        orders = points
    else:
        orders = [prev_point]+points

    idx2id = {i: oid for i, oid in enumerate(orders)}
    result_route_idxs, route_distance, _ = run_nlns_pipeline_lkh(matrix_to_solve, orders, model, device = 'cuda')
    basic_route = [idx2id[idx] for idx in result_route_idxs]
    start = basic_route[0]
    end = basic_route[-1]
    
    return basic_route, route_distance, start, end    

def solve_TSP_branch_and_cut(points, prev_point, matrix_to_solve, max_time_s=5):
    depot = 0

    n = len(points)
    if n == 0:
        return [], 0, None, None
    if n == 1:
        return [points[0]], 0, points[0], points[0]

    # Создаём solver (CBC)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    
    x = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x[i, j] = solver.BoolVar(f"x[{i},{j}]")

    # Не возвращаемся в depot: out = 1, in = 0
    solver.Add(solver.Sum(x[depot, j] for j in range(n) if j != depot) == 1)
    solver.Add(solver.Sum(x[i, depot] for i in range(n) if i != depot) == 0)

    # для остальных вершин: in = 1, out <= 1
    for i in range(n):
        if i == depot:
            continue
        solver.Add(solver.Sum(x[j, i] for j in range(n) if j != i) == 1)
        solver.Add(solver.Sum(x[i, j] for j in range(n) if j != i) <= 1)

    # минимизация суммарного времени
    solver.Minimize(solver.Sum(matrix_to_solve[i][j] * x[i, j]
                                for i in range(n) for j in range(n) if i != j))

    def extract_subtours_from_solution():
        """
        Возвращает список замкнутых циклов по текущему решению x
        """
        succ = [None] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = x[i, j].solution_value()
                if val is not None and val > 0.5:
                    succ[i] = j
                    break  # у каждой вершины максимум один исходящий путь
    
        visited_global = set()
        subtours = []
    
        for start in range(n):
            if start in visited_global:
                continue
    
            cur = start
            seen_local = set()
    
            while cur is not None:
                if cur in seen_local:
                    # нашли цикл — восстанавливаем его
                    cycle = []
                    node = cur
                    while node is not None and node not in cycle:
                        cycle.append(node)
                        node = succ[node]
                        if node == cur:
                            break
    
                    if cycle:
                        subtours.append(cycle)
                    break
    
                if cur in visited_global:
                    # Уже посещали эту вершину ранее — цикл не образуется
                    break
    
                seen_local.add(cur)
                visited_global.add(cur)
                cur = succ[cur]  # переход к следующему узлу
                
        return subtours


    status = None
    solver.SetTimeLimit(int(max_time_s * 1000))
    while True:
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            subtours = extract_subtours_from_solution()
            if len(subtours) == 0:
                break

            added = 0
            for S in subtours:
                if depot in S:
                    continue
                if len(S) <= 1:
                    continue
                solver.Add(solver.Sum(x[i, j] for i in S for j in S if i != j) <= len(S) - 1)
                added += 1

            if added == 0:
                break
            continue
        else:
            break
    # or status == pywraplp.Solver.FEASIBLE        
    if status == pywraplp.Solver.OPTIMAL: 
        succ = [-1] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = x[i, j].solution_value()
                if val is not None and val > 0.5:
                    succ[i] = j
                    break

        route_idx = []
        cur = depot
        route_idx.append(cur)
        visited = set([cur])
        while True:
            nxt = succ[cur]
            if nxt == -1 or nxt is None:
                break
            route_idx.append(nxt)
            if nxt in visited:
                # защита от зацикливания — остановимся
                break
            visited.add(nxt)
            cur = nxt
            if len(visited) >= n:
                break

        # путь не замыкаем
        # вычислим расстояние по ребрам пути
        total_distance = 0
        for k in range(len(route_idx) - 1):
            i = route_idx[k]
            j = route_idx[k + 1]
            total_distance += matrix_to_solve[i][j]

        route_points = [points[i] for i in route_idx]
        start_point = points[depot]
        end_point = points[route_idx[-1]]

        return route_points, total_distance, start_point, end_point

def solve_lkh_atsp(points, start, dist, runs=10):    
    if isinstance(dist, list):
        dist = np.array(dist)
    
    # Validate input
    if len(dist.shape) != 2:
        raise ValueError(f"Distance matrix must be 2D, got {dist.shape}")
    
    n = dist.shape[0]
    
    if n < 3:
        raise ValueError(f"LKH-3 requires at least 3 nodes, but got {n}")
    
    if dist.shape[0] != dist.shape[1]:
        raise ValueError(f"Distance matrix must be square, got {dist.shape}")

    modified_dist = dist.copy()
    BIG = 10**5

    for i in range(n):
        if i != start:
            modified_dist[i, start] = BIG

    uid = uuid.uuid4().hex
    lkh_dir = r"D:\LKHWin-3.0.13"
    lkh_executable = os.path.join(lkh_dir, "LKH-3.exe")

    tmp_dir = os.path.join(lkh_dir, f"tmp_{uid}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        tsp_path = os.path.join(tmp_dir, f"{uid}.tsp")
        par_path = os.path.join(tmp_dir, f"{uid}.par")
        tour_path = os.path.join(tmp_dir, f"{uid}.tour")

        # FIX 1: Strict TSPLIB formatting (no spaces before colons, standard '\n' line endings)
        with open(tsp_path, "w", encoding='ascii') as f:
            f.write(f"NAME: atsp_open\n")
            f.write(f"TYPE: ATSP\n")
            f.write(f"DIMENSION: {n}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write(f"EDGE_WEIGHT_SECTION\n")
            for i in range(n):
                # Ensure values are integers to prevent float formatting issues
                row = [str(int(modified_dist[i, j])) for j in range(n)]
                f.write(" ".join(row) + "\n")
            f.write("EOF\n")

        # FIX 2: Explicitly force PRECISION = 1 to bypass the buggy "too large" check 
        # in some LKH-3 Windows builds. It is ignored anyway for EXPLICIT matrices.
        with open(par_path, "w", encoding='ascii') as f:
            f.write(f"PROBLEM_FILE = {tsp_path}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
            f.write(f"RUNS = {runs}\n")
            f.write(f"PRECISION = 1\n")
            f.write(f"TRACE_LEVEL = 0\n")
            f.write("EOF\n")

        result = subprocess.run(
            [lkh_executable, par_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            cwd=lkh_dir
        )

        if result.returncode != 0:
            print("LKH-3 Error:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise subprocess.CalledProcessError(result.returncode, result.args)

        with open(tour_path, encoding='ascii') as f:
            lines = f.readlines()

        tour = []
        for line in lines:
            if line.strip().isdigit():
                v = int(line.strip()) - 1
                tour.append(v)

        if start in tour:
            idx = tour.index(start)
            tour = tour[idx:] + tour[:idx]

        lkh_route = [points[i] for i in tour]
        
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += dist[tour[i], tour[i + 1]]
            
        return lkh_route, total_distance

    finally:
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

def solve_TSP_Christofides_inner(points, prev_point, matrix_to_solve):
    n = len(points)
    if n == 0:
        return [], 0, None, None
    elif n == 1:
        single_point = points[0]
        return [single_point], 0, single_point, single_point

    added_prev = False
    if prev_point not in points:
        points = [prev_point] + points
        added_prev = True
    
    depot = 0
    
    manager = pywrapcp.RoutingIndexManager(
        len(matrix_to_solve), 1, depot
    )

    routing = pywrapcp.RoutingModel(manager)

    index_to_node_cache = {}
    
    for node in range(len(matrix_to_solve)):
        idx = manager.NodeToIndex(node)
        index_to_node_cache[idx] = node
    
    start_idx = routing.Start(0)
    index_to_node_cache[start_idx] = manager.IndexToNode(start_idx)
    
    end_idx = routing.End(0)
    index_to_node_cache[end_idx] = manager.IndexToNode(end_idx)

    def distance_callback(from_index, to_index, _dm=matrix_to_solve, _idx=index_to_node_cache, _dep=depot):
        from_node = _idx[from_index]
        to_node = _idx[to_index]
        return 0 if to_node == _dep else int(_dm[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES
    )

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        route_distance = 0
        
        if added_prev:
            index = solution.Value(routing.NextVar(index))
        
        start_node = index_to_node_cache[index]
        
        while not routing.IsEnd(index):
            node_index = index_to_node_cache[index]
            route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

        end_node = index_to_node_cache[previous_index]
        
        result_route = [points[i] for i in route]
        
        if added_prev:
            start_point = points[start_node] if start_node > 0 else None
            end_point = points[end_node] if end_node > 0 else None
        else:
            start_point = points[start_node]
            end_point = points[end_node]

        return result_route, route_distance, start_point, end_point


# ======================= фабрика для TAPOCT-FD =======================
def _matnet(points, prev_point, cost_matrix):
    """Интерфейс TAPOCT: (points, depot, matrix) -> (route, cost)."""
    if not points or points[0] != prev_point:
        raise ValueError("MatNet expects the depot at points[0]")

    model, model_device = _load_matnet()
    with torch.inference_mode():
        indices, cost = run_matnet_inference(
            np.asarray(cost_matrix),
            model,
            device=str(model_device),
            start_idx=0,
        )

    route = [points[int(i)] for i in indices]
    return route, float(cost)


def _nlns(points, prev_point, cost_matrix):
    """Интерфейс TAPOCT: (points, depot, matrix) -> (route, cost)."""
    if not points or points[0] != prev_point:
        raise ValueError("NLNS expects the depot at points[0]")

    model, model_device = _load_nlns()
    with torch.inference_mode():
        indices, cost, _ = run_nlns_pipeline_lkh(
            np.asarray(cost_matrix),
            points,
            model,
            start_idx=0,
            device=str(model_device),
        )

    route = [points[int(i)] for i in indices]
    return route, float(cost)


def _lkh3(points, prev_point, cost_matrix, runs=5):
    """(list[id] n+1, id, (n+1,n+1)) -> (list[id], float).
    В оригинале в solve_lkh_atsp передавался prev_point как ИНДЕКС старта; это работало только
    потому, что id депо == 0 == его индекс. Здесь индекс 0 передаётся явно."""
    route, dist = solve_lkh_atsp(points, 0, np.asarray(cost_matrix), runs=runs)
    return route, float(dist)


def _christofides(points, prev_point, cost_matrix):
    """(list[id] n+1, id, (n+1,n+1)) -> (list[id], float)"""
    out = solve_TSP_Christofides_inner(points, prev_point, np.asarray(cost_matrix))
    if out is None:
        return None, None
    route, dist, _, _ = out
    return route, float(dist)


def get_solvers():
    """-> dict[str, SolverFn] в порядке классов MLP: MatNet, NLNS, lkh3, Christofides"""
    return {"MatNet": _matnet, "NLNS": _nlns, "lkh3": _lkh3, "Christofides": _christofides}
