"""
SpaceAgent: A Spatially-Aware LLM Framework for Gene Set Analysis.
Target: ACL / NeurIPS / ISMB / Bioinformatics
Author: SpaceAgent Team
License: MIT
"""

import os
import json
import logging
import requests
import warnings
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Union, Any

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import scipy.spatial
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline
from scipy.stats import ttest_ind
import random
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects


# ==========================================
# 5. 高级评估器 (新增 Neighbor_Corr 指标)
# ==========================================
# ==========================================
# 5. 高级评估器 (微环境特异性版)
# ==========================================
# ==========================================
# 5. 高级评估器 (异质邻域特异性版)
# ==========================================
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedEvaluator:
    def __init__(self, adata, config):
        self.adata = adata
        self.cfg = config
        
        # 加载功能通路数据库 (用于评估 Func_Jaccard)
        self.gene_to_pathways = self._load_db()
        
        # [核心修改] 构建基于物理半径的空间邻接矩阵
        self.adj_matrix = self._compute_spatial_connectivity()

    def _load_db(self):
        """加载并解析 KEGG 数据库 (保持原逻辑)"""
        if not os.path.exists(self.cfg.kegg_path):
            logger.info("⬇️ Downloading KEGG database...")
            try:
                r = requests.get("https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human", timeout=30)
                if r.status_code == 200:
                    with open(self.cfg.kegg_path, 'w', encoding='utf-8') as f:
                        f.write(r.text)
            except: return {}

        mapping = {}
        try:
            with open(self.cfg.kegg_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    term = parts[0]
                    for g in parts[2:]:
                        g_upper = g.upper()
                        if g_upper not in mapping: mapping[g_upper] = set()
                        mapping[g_upper].add(term)
            logger.info(f"✅ Functional DB Loaded: {len(mapping)} genes covered.")
            return mapping
        except: return {}

    def _compute_spatial_connectivity(self):
        """
        [升级版] 计算空间连接性权重矩阵 (Distance-Weighted)
        逻辑：不再只返回 0/1，而是返回基于高斯核或线性衰减的权重。
        """
        if 'spatial_norm' in self.adata.obsm:
            coords = self.adata.obsm['spatial_norm']
            radius = getattr(self.cfg, 'adaptive_threshold', 0.05)
        else:
            coords = self.adata.obsm['spatial']
            radius = 250.0 
        
        # 1. 计算距离矩阵 (mode='distance')
        nbrs = NearestNeighbors(radius=radius).fit(coords)
        dists = nbrs.radius_neighbors_graph(coords, mode='distance')
        
        # 2. 将距离转换为权重 (高斯核衰减)
        # 距离越近，权重越接近 1；距离接近 radius，权重接近 0
        # 公式: w = exp(- (dist^2) / (2 * (radius/3)^2))  # 假设 radius 是 3 sigma
        
        # 注意：radius_neighbors_graph 返回的是稀疏矩阵
        # 我们直接操作其 data 属性
        sigma = radius / 3.0 # 经验值
        dists.data = np.exp(-(dists.data**2) / (2 * sigma**2))
        
        # 3. 移除自环 (对角线)，因为我们关注的是 Neighbor 交互
        dists.setdiag(0)
        dists.eliminate_zeros()
        
        logger.info(f"⚖️ Evaluator: Weighted Spatial Graph built (Radius={radius:.4f}, Sigma={sigma:.4f})")
        return dists

    def calc_functional_score(self, ga, gb):
        """计算功能通路 Jaccard 相似度"""
        if not self.gene_to_pathways: return 0.0
        pa = self.gene_to_pathways.get(ga, set())
        pb = self.gene_to_pathways.get(gb, set())
        union = len(pa.union(pb))
        return len(pa.intersection(pb)) / union if union > 0 else 0.0

    def calc_spatial_metrics(self, ga, gb, cluster_id):
        """
        计算空间一致性指标 (升级版)
        """
        try:
            # 1. 获取基因索引与表达向量
            idx_a = self.adata.var_names.get_loc(ga)
            idx_b = self.adata.var_names.get_loc(gb)
            
            # 兼容稀疏矩阵提取
            get_val = lambda idx: self.adata.X[:, idx].toarray().flatten() if sparse.issparse(self.adata.X) else self.adata.X[:, idx].flatten()
            va_full = get_val(idx_a) # 配体表达量 (全组织)
            vb_full = get_val(idx_b) # 受体表达量 (全组织)
            
            # 2. 锁定当前 Cluster 的细胞
            # 使用 astype(str) 确保 ID 匹配准确
            mask_self = self.adata.obs['target_cluster'].astype(str) == str(cluster_id)
            if mask_self.sum() < 5: return 0.0, 0.0 # 细胞太少不予评估

            # ---------------------------------------------------------
            # 指标 A: Local Score (Same Spot / Autocrine)
            # 逻辑：衡量同一位置（或同一 Cluster 内）的共表达强度
            # ---------------------------------------------------------
            va_local = va_full[mask_self]
            vb_local = vb_full[mask_self]
            
            norm_a = np.linalg.norm(va_local)
            norm_b = np.linalg.norm(vb_local)
            
            if norm_a > 0 and norm_b > 0:
                score_local = np.dot(va_local, vb_local) / (norm_a * norm_b)
            else:
                score_local = 0.0

            # ---------------------------------------------------------
            # 指标 B: Neighbor Score (Paracrine / Heterotypic)
            # 逻辑：衡量 Cluster 边缘细胞与“非同类”邻居之间的通讯强度
            # ---------------------------------------------------------
            
            # 取出当前 Cluster 对应行的邻接关系 (Rows: Cluster Cells, Cols: All Cells)
            sub_adj = self.adj_matrix[mask_self, :].copy()
            
            # 【关键】将通向 Cluster 内部（自己人）的连线全部置零
            # 我们只关心: Cluster Cell -> External Neighbor
            sub_adj[:, mask_self] = 0 
            sub_adj.eliminate_zeros() # 清理零值，保持稀疏性
            
            # 统计每个细胞有多少个“异质邻居”
            weights = sub_adj.sum(axis=1).A.flatten()
            valid_border_cells = weights > 0 # 只有位于边界的细胞才有异质邻居
            
            if valid_border_cells.sum() < 3: 
                # 如果完全没有边界（被包裹或独立），则无旁分泌
                score_neighbor = 0.0
            else:
                # 计算异质邻居的 B 基因平均表达量 (加权平均)
                # vb_neighbor_hetero[i] = (Neighbors of i) * Expression_B / Count
                vb_neighbor_hetero = sub_adj.dot(vb_full)
                vb_neighbor_hetero[valid_border_cells] /= weights[valid_border_cells]
                
                # 计算 Cosine Similarity:
                # 向量1: 边界细胞的 A 表达量
                # 向量2: 对应异质邻居的 B 平均表达量
                va_border = va_local[valid_border_cells]
                vb_border_neighbor = vb_neighbor_hetero[valid_border_cells]
                
                n_a_border = np.linalg.norm(va_border)
                n_b_neighbor = np.linalg.norm(vb_border_neighbor)
                
                if n_a_border > 0 and n_b_neighbor > 0:
                    score_neighbor = np.dot(va_border, vb_border_neighbor) / (n_a_border * n_b_neighbor)
                else:
                    score_neighbor = 0.0

            return float(score_local), float(score_neighbor)
            
        except Exception as e: 
            # logging.warning(f"Metric calc failed for {ga}-{gb}: {e}")
            return 0.0, 0.0

    def run_benchmark(self, results):
        logger.info("⚖️ Running Quantitative Benchmark...")
        stats = []
        
        for model in ["GeneAgent", "SpaceAgent"]:
            for entry in results:
                cid = entry['cluster_id']
                llm_out = entry.get('llm_analysis', '')
                
                # 对于 SpaceAgent，如果 LLM 没有输出分析，跳过
                if model == "SpaceAgent" and not llm_out: continue
                
                for pair in entry.get('known_interactions', []):
                    try:
                        g1, g2 = pair.split(' -> ')
                        
                        # === SpaceAgent 的严格过滤逻辑 ===
                        if model == "SpaceAgent":
                            # 1. 基因必须出现在文本中
                            if (g1 not in llm_out) or (g2 not in llm_out):
                                continue
                            
                            # 2. 必须没有被标记为负面结果
                            # (简单的关键词匹配，实际可用正则增强)
                            if "Segregated" in llm_out or "Silent" in llm_out:
                                continue
                        # ==================================
                        
                        # 计算三项指标
                        f_score = self.calc_functional_score(g1, g2)
                        s_local, s_neighbor = self.calc_spatial_metrics(g1, g2, cluster_id=cid)
                        
                        if s_local > 0 or s_neighbor > 0:
                            stats.append({
                                "Model": model, 
                                "Cluster": cid,
                                "Pair": pair,
                                "Func_Jaccard": f_score, 
                                "Same_Spot_Score": s_local, 
                                "Neighbor_Score": s_neighbor
                            })
                    except: pass
        
        df = pd.DataFrame(stats)
        if not df.empty:
            print("\n" + "="*60)
            print("📊 Benchmark Summary (Mean Scores):")
            print(df.groupby("Model")[["Func_Jaccard", "Same_Spot_Score", "Neighbor_Score"]].mean())
            print("=" * 60)
            return df
        else:
            logger.warning("⚠️ No valid interactions found for benchmarking.")
            return None
    
# ==========================================
# [New Class] 幻觉分析器 (HallucinationAnalyzer)
# ==========================================
class HallucinationAnalyzer:
    def __init__(self, space_analyzer):
        self.sa = space_analyzer

    def collect_distances(self, adata, results_json) -> Dict[str, List[float]]:
        dist_data = {"Random": [], "MyAgent": [], "SpaceAgent": []}
        logger.info("📉 Calculating Spatial Distance Distributions...")
        
        for entry in results_json:
            cid = str(entry['cluster_id'])
            llm_out = entry.get('llm_analysis', '')
            
            # 1. 收集 MyAgent 和 SpaceAgent
            if 'known_interactions' in entry:
                for pair in entry['known_interactions']:
                    try:
                        ga, gb = pair.split(' -> ')
                        dist = self.sa.verify_interaction(adata, ga, gb, cid)
                        if isinstance(dist, float) and dist >= 0:
                            dist_data["MyAgent"].append(dist)
                            # 宽松条件：只要文中提到了这两个基因，且没有明确拒绝
                            if (ga in llm_out) and (gb in llm_out) and ("Segregated" not in llm_out):
                                dist_data["SpaceAgent"].append(dist)
                    except: pass

            # 2. 收集 Random
            if 'gene_list' in entry:
                genes = entry['gene_list']
                if len(genes) > 2:
                    for _ in range(15):
                        try:
                            g1, g2 = random.sample(genes, 2)
                            dist = self.sa.verify_interaction(adata, g1, g2, cid)
                            if isinstance(dist, float) and dist >= 0:
                                dist_data["Random"].append(dist)
                        except: pass
        
        # --- 关键修复：智能自适应过滤 ---
        all_values = dist_data["Random"] + dist_data["MyAgent"]
        if all_values:
            # 计算 98% 分位数
            limit = np.percentile(all_values, 98)
            
            # 【修复点】如果大部分数据都是 0 (limit=0)，则强制给一个合理的观测窗口
            # 或者使用最大值，防止数据被清空
            if limit < 1.0:
                logger.warning(f"   ⚠️ Data is highly co-localized (98% are 0.0). Using max value as limit.")
                limit = np.max(all_values)
                # 如果最大值还是 0，强制给一个绘图范围 (例如 100.0) 以便画出坐标轴
                if limit < 1.0: limit = 100.0
            
            logger.info(f"   Auto-detected distance limit: {limit:.1f} (pixels/units)")
            
            for k in dist_data:
                # 过滤掉极端离群值，但保留 0
                dist_data[k] = [x for x in dist_data[k] if x <= limit]
        
        logger.info(f"   Samples collected - Random: {len(dist_data['Random'])}, "
                    f"MyAgent: {len(dist_data['MyAgent'])}, "
                    f"SpaceAgent: {len(dist_data['SpaceAgent'])}")
        return dist_data
    
# ==========================================
# 6. 可视化工具 (Visualizer)
# ==========================================
# ==========================================
# 6. 可视化工具 (Visualizer) - 修复版
# ==========================================
# 计算平均最近邻距离作为直径
from sklearn.neighbors import NearestNeighbors

def get_density_based_size(adata):
    coords = adata.obsm['spatial']
    # 随机采样一部分点计算平均距离，避免大数据量卡顿
    sample_idx = np.random.choice(coords.shape[0], min(1000, coords.shape[0]), replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(coords[sample_idx])
    distances, _ = nn.kneighbors(coords[sample_idx])
    return np.median(distances[:, 1]) * 0.5 # 取平均邻居距离的 80%

class Visualizer:
    @staticmethod
    def plot_interaction_spatial(adata: sc.AnnData, 
                               entry: Dict, 
                               save_path: str = "interaction_plot.pdf"):
        import matplotlib.pyplot as plt
        
        # 1. 准备数据
        target_pair = None
        analysis_text = entry.get('llm_analysis', 'No analysis.')
        known = entry.get('known_interactions', [])
        
        for pair in known:
            g1, g2 = pair.split(' -> ')
            if g1 in analysis_text and g2 in analysis_text:
                target_pair = (g1, g2)
                break
        if not target_pair and known: target_pair = known[0].split(' -> ')
        if not target_pair: return

        ga, gb = target_pair
        cid = entry['cluster_id']
        
        # 2. 检查 2D 还是 3D
        is_3d = adata.obsm['spatial'].shape[1] == 3
        
        # 3. 设置绘图
        # 如果是 3D，我们不能用 sc.pl.spatial，改用 matplotlib 3D scatter
        if is_3d:
            Visualizer._plot_3d(adata, ga, gb, cid, analysis_text, save_path)
        else:
            Visualizer._plot_2d(adata, ga, gb, cid, analysis_text, save_path)

    @staticmethod
    def _plot_2d(adata, ga, gb, cid, text, save_path):
        import matplotlib.pyplot as plt
        
        # 尝试自动推断 spot_size
        # seqFISH 等单细胞数据点很小，Visium 点很大
        # 这里给一个启发式默认值，或者捕获异常
        # 在 _plot_2d 内部

        spot_size = get_density_based_size(adata)
            
        fig, axs = plt.subplots(1, 5, figsize=(24, 5), constrained_layout=True)
        
        try:
            # 封装绘图调用，处理 spot_size 报错
            def safe_spatial(color, ax, title, cmap=None, groups=None):
                try:
                    sc.pl.spatial(adata, color=color, ax=ax, show=False, title=title, 
                                  cmap=cmap, groups=groups, use_raw=False, spot_size=spot_size)
                except ValueError:
                    # 如果报错 (spot_size required)，尝试硬编码一个值重试
                    sc.pl.spatial(adata, color=color, ax=ax, show=False, title=title, 
                                  cmap=cmap, groups=groups, use_raw=False, spot_size=spot_size)

            # A. Tissue
            safe_spatial(None, axs[0], "Tissue")
            
            # B. Cluster
            cluster_col = adata.obs['target_cluster'].name
            safe_spatial(cluster_col, axs[1], f"Cluster: {cid}", groups=[cid])
            axs[1].legend().remove()
            
            # C. Ligand & Receptor
            safe_spatial(ga, axs[2], f"Ligand: {ga}", cmap='Reds')
            safe_spatial(gb, axs[3], f"Receptor: {gb}", cmap='Blues')
            
            # D. Text
            axs[4].axis('off')
            axs[4].text(0, 1, f"Report:\n{ga}->{gb}\n\n{text[:600]}...", va='top', wrap=True)
            
            plt.savefig(save_path)
            logging.info(f"✅ Figure saved: {save_path}")
            plt.close()
            
        except Exception as e:
            logging.error(f"❌ 2D Plotting failed: {e}")

    @staticmethod
    def _plot_3d(adata, ga, gb, cid, text, save_path):
        """专门处理 3D 数据的绘图"""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(24, 6))
        coords = adata.obsm['spatial']
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        def add_scatter(idx, color_data, title, cmap=None):
            ax = fig.add_subplot(1, 5, idx, projection='3d')
            # 简单的 3D 散点
            if cmap:
                p = ax.scatter(x, y, z, c=color_data, cmap=cmap, s=2, alpha=0.6)
                plt.colorbar(p, ax=ax, shrink=0.5)
            else:
                # 离散颜色 (Cluster)
                # 简单处理：只高亮目标 cluster
                mask = adata.obs['target_cluster'] == cid
                ax.scatter(x[~mask], y[~mask], z[~mask], c='lightgrey', s=1, alpha=0.1)
                ax.scatter(x[mask], y[mask], z[mask], c='red', s=5, label=cid)
                if idx == 2: ax.legend()
            
            ax.set_title(title)
            ax.set_axis_off()

        # 1. Overview (全灰)
        ax1 = fig.add_subplot(1, 5, 1, projection='3d')
        ax1.scatter(x, y, z, c='grey', s=1, alpha=0.1)
        ax1.set_title("3D Tissue")
        ax1.set_axis_off()
        
        # 2. Cluster
        add_scatter(2, None, f"Cluster: {cid}")
        
        # 3. Ligand
        val_a = adata[:, ga].X.toarray().flatten() if sparse.issparse(adata.X) else adata[:, ga].X.flatten()
        add_scatter(3, val_a, f"Ligand: {ga}", 'Reds')
        
        # 4. Receptor
        val_b = adata[:, gb].X.toarray().flatten() if sparse.issparse(adata.X) else adata[:, gb].X.flatten()
        add_scatter(4, val_b, f"Receptor: {gb}", 'Blues')
        
        # 5. Text
        ax5 = fig.add_subplot(1, 5, 5)
        ax5.axis('off')
        ax5.text(0, 1, f"3D Analysis:\n{ga}->{gb}\n\n{text[:600]}...", va='top', wrap=True)
        
        plt.savefig(save_path)
        logging.info(f"✅ 3D Figure saved: {save_path}")
        plt.close()

    @staticmethod
    def plot_benchmark_metrics(df_stats, save_path="benchmark_metrics.pdf"):
        # (保持不变)
        import matplotlib.pyplot as plt
        import seaborn as sns
        if df_stats is None or df_stats.empty: return
        try:
            df_long = df_stats.melt(id_vars=["Model"], 
                                   value_vars=["Func_Jaccard", "Same_Spot_Corr", "Neighbor_Corr"],
                                   var_name="Metric", value_name="Score")
            plt.figure(figsize=(10, 6), dpi=150)
            sns.set_style("whitegrid")
            palette = {"GeneAgent": "#E74C3C", "SpaceAgent": "#3498DB"}
            ax = sns.barplot(data=df_long, x="Metric", y="Score", hue="Model", 
                             palette=palette, errorbar="se", capsize=0.1)
            plt.title("Performance Benchmark", fontsize=14)
            plt.savefig(save_path)
            plt.close()
        except Exception: pass

    @staticmethod
    def plot_hallucination_test(dist_data: Dict[str, List[float]], save_path: str = "Figure_4_Hallucination.pdf"):
        """
        绘制空间距离分布对比图 (修复版：强制非负，解决负值问题)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import ttest_ind
        
        if not any(dist_data.values()):
            logging.error("❌ No distance data to plot.")
            return

        plt.figure(figsize=(10, 6), dpi=150)
        sns.set_style("whitegrid")
        
        has_plot = False
        colors = {"Random": "grey", "MyAgent": "#E74C3C", "SpaceAgent": "#3498DB"}
        labels = {"Random": "Random Pairs", "MyAgent": "MyAgent (DB)", "SpaceAgent": "SpaceAgent (Ours)"}
        
        for key in ["Random", "MyAgent", "SpaceAgent"]:
            data = dist_data[key]
            if len(data) > 2:
                # 检查低方差情况
                if np.std(data) < 1e-6:
                    # 【修复点1】改用正向抖动 (Uniform 0~0.1)，保证不出现负数
                    jittered_data = np.array(data) + np.random.uniform(0, 0.1, len(data))
                    # 【修复点2】clip=(0, None) 强制截断，禁止画到 0 左边
                    sns.kdeplot(jittered_data, color=colors[key], label=labels[key], 
                              fill=True, alpha=0.1, clip=(0, None))
                else:
                    # 普通数据也加上 clip=(0, None)
                    sns.kdeplot(data, color=colors[key], label=labels[key], 
                              fill=(key=="Random"), alpha=0.1, clip=(0, None))
                has_plot = True

        if not has_plot:
            logging.error("❌ Not enough data points to estimate KDE.")
            plt.close()
            return

        title_text = "Spatial Distance Distribution Check"
        if len(dist_data["SpaceAgent"]) > 1 and len(dist_data["MyAgent"]) > 1:
            try:
                if np.std(dist_data["SpaceAgent"]) > 0 or np.std(dist_data["MyAgent"]) > 0:
                    t, p = ttest_ind(dist_data["SpaceAgent"], dist_data["MyAgent"], equal_var=False)
                    title_text += f"\n(SpaceAgent vs MyAgent: p={p:.2e})"
            except: pass

        plt.title(title_text, fontsize=14, fontweight='bold')
        plt.xlabel("Physical Distance (µm approx.)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.xlim(0, None) # 强制X轴从0开始
        
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"✅ Hallucination plot saved: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_lr_colocalization_heatmap(adata, entry: Dict, save_path: str = "lr_halo_plot.pdf"):
        """
        [NEW] 绘制 "Halo" 图：Ligand * Receptor 活性乘积热图
        展示空间上的互作热点 (Interaction Hotspots)
        """
        import matplotlib.pyplot as plt
        
        # 1. 确定目标基因对
        target_pair = None
        analysis_text = entry.get('llm_analysis', '')
        known = entry.get('known_interactions', [])
        for pair in known:
            g1, g2 = pair.split(' -> ')
            # 优先画 LLM 认为 Valid 的对
            if g1 in analysis_text and g2 in analysis_text and "Segregated" not in analysis_text:
                target_pair = (g1, g2)
                break
        if not target_pair and known: target_pair = known[0].split(' -> ')
        if not target_pair: return

        ga, gb = target_pair
        cid = entry['cluster_id']
        
        # 2. 计算互作得分 (Element-wise Product)
        try:
            idx_a = adata.var_names.get_loc(ga)
            idx_b = adata.var_names.get_loc(gb)
            get_val = lambda idx: adata.X[:, idx].toarray().flatten() if sparse.issparse(adata.X) else adata.X[:, idx].flatten()
            
            # 归一化以避免数值差异过大
            va = get_val(idx_a)
            vb = get_val(idx_b)
            va = (va - va.min()) / (va.max() - va.min() + 1e-9)
            vb = (vb - vb.min()) / (vb.max() - vb.min() + 1e-9)
            
            # 核心公式：L * R
            interaction_score = va * vb
            
            # 3. 绘图
            fig, ax = plt.subplots(figsize=(8, 8))
            coords = adata.obsm['spatial']
            
            # 背景灰点 (Tissue)
            ax.scatter(coords[:, 0], coords[:, 1], c='lightgrey', s=1, alpha=0.3)
            
            # 前景热点 (Interaction)，只画分数 > 0 的点
            mask = interaction_score > 0.01
            if mask.sum() > 0:
                sc = ax.scatter(coords[mask, 0], coords[mask, 1], 
                                c=interaction_score[mask], 
                                cmap='magma', s=20, alpha=0.8, edgecolors='none')
                plt.colorbar(sc, ax=ax, label=f"Interaction Score ({ga}-{gb})")
            
            ax.set_title(f"Spatial Interaction Hotspots: {ga} -> {gb}\n(Cluster {cid})")
            ax.axis('off')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"✅ Halo Plot saved: {save_path}")
            plt.close()
            
        except Exception as e:
            logging.error(f"❌ Halo Plot failed: {e}")

# ==========================================
# 5.5. 质量评估器 (QualityEvaluator - LLM-as-a-Judge)
# ==========================================
# ==========================================
# 5.5. 质量评估器 (Revised with GPT-4 Support)
# ==========================================
class QualityEvaluator:
    def __init__(self, config: SpaceConfig):
        self.cfg = config
        self.pipe = None
        self.client = None

    def load_judge_model(self):
        """Loads Judge Model (Follows global LLM config or defaults to API for better judging)"""
        
        # NOTE: You might want to force API for judging even if using local for generation
        # For now, we follow the self.cfg.llm_source logic
        
        if self.cfg.llm_source == "api":
            logger.info(f"⚖️ Initializing GPT-4 as Judge...")
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.cfg.openai_api_key,
                    base_url=self.cfg.openai_base_url
                )
            except Exception as e: logger.error(e)
        else:
            logger.info(f"⚖️ Initializing Local Judge ({self.cfg.model_id})...")
            try:
                self.pipe = pipeline(
                    "text-generation",
                    model=self.cfg.model_id,
                    device=self.cfg.gpu_id,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
            except Exception as e:
                logger.error(f"⚠️ Judge model failed: {e}")

    def _generate_judge_response(self, prompt_messages: List[Dict]) -> str:
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.openai_model,
                    messages=prompt_messages,
                    temperature=0.0, # Judge should be deterministic
                    max_tokens=10
                )
                return response.choices[0].message.content
            except: return ""
        elif self.pipe:
            try:
                out = self.pipe(prompt_messages, max_new_tokens=10, do_sample=False)
                return out[0]["generated_text"][-1]["content"]
            except: return ""
        return ""

    def generate_mock_baseline(self, entry: Dict) -> str:
        # ... (Keep existing implementation) ...
        genes = ", ".join(entry.get('gene_list', [])[:10])
        pairs = ", ".join(entry.get('known_interactions', [])[:5])
        return (
            f"Analysis for Cluster {entry.get('cluster_id')}:\n"
            f"Based on the OmniPath database, we identified potential ligand-receptor interactions "
            f"among the top expressed genes ({genes}). \n"
            f"Key interactions include: {pairs}. \n"
            f"These interactions are known to be involved in general cell signaling pathways according to KEGG references."
        )

    def _extract_score(self, text: str) -> float:
        # ... (Keep existing implementation) ...
        import re
        matches = re.findall(r"Score:\s*([\d\.]+)", text, re.IGNORECASE)
        if matches:
            try:
                score = float(matches[0])
                return min(max(score, 1.0), 10.0)
            except: pass
        return 5.0

    def evaluate_fairness(self, results: List[Dict]) -> pd.DataFrame:
        if not self.pipe and not self.client:
            self.load_judge_model()
            
        logger.info(f"⚖️ Running Multi-Dimensional Scoring on {len(results)} clusters...")
        scores = []
        eval_targets = results 
        
        for i, entry in enumerate(eval_targets):
            report_base = self.generate_mock_baseline(entry)
            report_ours = entry.get('llm_analysis', 'No analysis')
            
            if len(report_ours) < 50: continue

            row = {"Cluster": entry['cluster_id']}
            
            dimensions = [
                ("Biological Plausibility", "Does the logic make biological sense?"),
                ("Sample Specificity", "Is the analysis specific to this tissue's spatial context?"),
                ("Scientific Trustworthiness", "Is the report free from hallucinations and over-claims?")
            ]
            
            print(f"   Judge Processing {i+1}/{len(eval_targets)}...")
            
            for dim_name, dim_desc in dimensions:
                for model_name, report_text in [("MyAgent", report_base), ("SpaceAgent", report_ours)]:
                    
                    prompt = [
                        {"role": "system", "content": "You are a critical scientific reviewer. Rate the analysis report on a scale of 1 to 10. Be strict but fair."},
                        {"role": "user", "content": f"""
Task: Evaluate the report based on: **{dim_name}** ({dim_desc}).

[Report Start]
{report_text}
[Report End]

Provide a single number score (1-10) after 'Score:'.
Analysis:
"""}
                    ]
                    
                    response = self._generate_judge_response(prompt)
                    score = self._extract_score(response)
                    
                    key = f"{model_name}_{dim_name.split()[0]}" 
                    row[key] = score
                        
            scores.append(row)
            
        df = pd.DataFrame(scores)
        if not df.empty:
            print("\n" + "="*70)
            print("📊 Table 3: Full Multi-Dimensional Evaluation (Average Scores)")
            print("="*70)
            cols = [c for c in df.columns if c != "Cluster"]
            summary = df[cols].mean().to_frame(name="Avg Score")
            print(summary)
            print("="*70)
        return df



# ==========================================
# 7. 高级可视化引擎 (Publication-Ready V2)
# ==========================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from scipy import sparse
import logging

logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    def __init__(self, adata, config):
        self.adata = adata
        self.cfg = config
        self.coords = adata.obsm['spatial']
        # 强制 Cluster ID 为字符串
        if 'target_cluster' in adata.obs:
            self.adata.obs['target_cluster'] = self.adata.obs['target_cluster'].astype(str)

    # =========================================================
    #  核心修复: 坐标对齐 + 暴力增大点大小
    # =========================================================
    
    def _get_scale_factor(self):
        """获取 Visium 的缩放因子 (Fullres -> Hires)"""
        try:
            if 'spatial' in self.adata.uns:
                lib_id = list(self.adata.uns['spatial'].keys())[0]
                scalefactors = self.adata.uns['spatial'][lib_id]['scalefactors']
                return scalefactors['tissue_hires_scalef']
        except Exception:
            pass
        return 1.0

    def _get_dynamic_marker_size(self, coords):
        """
        [重点修改] 根据坐标跨度动态计算点的大小
        大幅增加了返回的数值，确保在 Light Mode 下清晰可见
        """
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        max_span = max(x_range, y_range)
        
        # 逻辑：s 是面积 (points^2)。
        # 如果是像素级大坐标 (>1000)，如 Visium:
        if max_span > 1000: 
            return 100  # 之前是 80，扩大5倍 (直径约20点)
        # 如果是归一化坐标 (<5):
        elif max_span < 5:
            return 30  # 之前是 20，扩大6倍
        # 中间情况:
        else:
            return 60  # 之前是 40

    def _get_expr(self, gene):
        """安全获取归一化表达量"""
        try:
            idx = self.adata.var_names.get_loc(gene)
            if sparse.issparse(self.adata.X):
                val = self.adata.X[:, idx].toarray().flatten()
            else:
                val = self.adata.X[:, idx].flatten()
            if val.max() > 0:
                val = (val - val.min()) / (val.max() - val.min())
            return val
        except KeyError:
            return np.zeros(self.adata.n_obs)

    # =========================================================
    #  绘图功能 1: 聚类交互矩阵
    # =========================================================
    # 在 AdvancedVisualizer 类中
    def plot_cluster_interaction_matrix(self, results_df, value_col='strength', save_path=None):
        """
        绘制簇间交互热图
        修改：默认 value_col 改为 'strength' (加权强度)，而非 'count'
        """
        print(f"🔥 Plotting Cluster Interaction Matrix using column: {value_col}...")
        
        if results_df is None or results_df.empty:
            logger.warning("⚠️ Warning: No results to plot matrix.")
            return None
            
        if value_col not in results_df.columns:
            logger.warning(f"⚠️ Column '{value_col}' not found. Falling back to 'count'.")
            value_col = 'count'
            
        # 聚合：同一个 Source-Target 可能有多个配受体对，强度累加
        df_agg = results_df.groupby(['source', 'target'])[value_col].sum().reset_index()
        
        matrix = df_agg.pivot(index='source', columns='target', values=value_col).fillna(0)
        
        plt.figure(figsize=(11, 9)) # 稍微加大一点
        sns.set_style("white")
        
        # 使用更适合连续值的 cmap (如 Magma 或 Viridis)
        # 如果数值跨度大，考虑 log1p 处理: np.log1p(matrix)
        ax = sns.heatmap(
            matrix, 
            cmap="RdBu_r", 
            center=0, 
            annot=True, 
            fmt=".1f", # 保留一位小数即可
            linewidths=0.5, 
            linecolor='#f0f0f0', 
            square=True,
            cbar_kws={"shrink": 0.8, "label": f"Interaction {value_col.title()} (L*R * DistFactor)"}
        )
        
        plt.title("Cluster Interaction Strength Matrix (Mass Action Model)", fontsize=15, pad=20)
        plt.xlabel("Target Cluster (Receiver)", fontsize=12)
        plt.ylabel("Source Cluster (Sender)", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Interaction Matrix saved: {save_path}")
        
        plt.close()
        return matrix

    # =========================================================
    #  绘图功能 2: 空间网络图 (Light Mode) - 重点修复
    # =========================================================

    # =========================================================
    #  绘图功能 2: 空间网络图 (Light Mode) - [美化版]
    # =========================================================

    def plot_spatial_network_light(self, interaction_matrix, save_path, 
                                background_type='celltype', transparency=0.7):
        """
        [美化最终版] 亮色背景空间网络图
        改进点：
        1. 颜色分离：背景点使用 Pastel 色系，网络节点使用 Vivid (tab20) 色系。
        2. 线条优化：线条不使用灰色，而是跟随 Source 节点的颜色，且清晰可见。
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.patheffects as path_effects
        
        # 1. 坐标缩放
        scale_factor = 1.0
        if background_type == 'image':
            scale_factor = self._get_scale_factor()
        display_coords = self.coords * scale_factor
        
        # 2. 计算质心
        centroids = {}
        all_nodes = set(interaction_matrix.index).union(set(interaction_matrix.columns))
        unique_clusters = sorted(list(all_nodes))
        for c in unique_clusters:
            mask = self.adata.obs['target_cluster'].astype(str) == str(c)
            if mask.sum() > 0:
                centroids[str(c)] = np.mean(display_coords[mask], axis=0)
        
        # 3. 颜色管理 (核心修改)
        all_cluster_ids = sorted(self.adata.obs['target_cluster'].astype(str).unique())
        n_clusters = len(all_cluster_ids)
        
        # A. 节点颜色 (Node Color): 使用鲜艳的 tab20
        node_cmap = cm.get_cmap('tab20', n_clusters)
        node_color_map = {cid: node_cmap(i) for i, cid in enumerate(all_cluster_ids)}
        
        # B. 背景颜色 (Background Color): 使用淡雅的 Set3 或 Pastel1，与节点区分开
        bg_cmap = cm.get_cmap('Set3', n_clusters) 
        bg_color_map = {cid: bg_cmap(i) for i, cid in enumerate(all_cluster_ids)}

        # 4. 初始化画布
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 12)) 
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # 5. 绘制背景 (使用 bg_color_map)
        if background_type == 'image' and 'spatial' in self.adata.uns:
            self._plot_visium_image_aligned(ax)
        else:
            s_size = self._get_dynamic_marker_size(display_coords)
            if 'target_cluster' in self.adata.obs:
                # 使用淡色系绘制背景，且透明度较低，避免抢眼
                bg_colors = [bg_color_map.get(str(c), 'lightgrey') for c in self.adata.obs['target_cluster']]
                ax.scatter(display_coords[:, 0], display_coords[:, 1], 
                         c=bg_colors, s=s_size, 
                         linewidths=0, edgecolors='none', 
                         alpha=0.4, zorder=1) # 背景淡一点
            else:
                ax.scatter(display_coords[:, 0], display_coords[:, 1], c='#eeeeee', s=s_size, zorder=1)

        # 6. 构建网络
        G = nx.Graph()
        max_weight = interaction_matrix.values.max() if not interaction_matrix.empty else 1
        threshold = max_weight * 0.05
        
        for src in interaction_matrix.index:
            for tgt in interaction_matrix.columns:
                w = interaction_matrix.loc[src, tgt]
                if w > threshold and str(src) in centroids and str(tgt) in centroids:
                    if G.has_edge(str(src), str(tgt)): 
                        G[str(src)][str(tgt)]['weight'] += w
                    else: 
                        G.add_edge(str(src), str(tgt), weight=w)
        
        # 7. 绘制网络元素
        if len(G.nodes()) > 0:
            pos = {n: centroids[n] for n in G.nodes()}
            edges = list(G.edges(data=True))
            weights = [d['weight'] for _, _, d in edges]
            max_w = max(weights) if weights else 1
            
            # --- A. 绘制彩色连线 (核心修改) ---
            if weights:
                for (u, v, d) in edges:
                    w = d['weight']
                    width = 1.5 + (w / max_w) * 6.0
                    
                    # 获取源节点颜色用于连线
                    edge_color = node_color_map.get(str(u), 'grey')
                    
                    # 设定不透明度：不想太透明，所以最低 0.6，最高 0.9
                    alpha_val = 0.6 + (w / max_w) * 0.3
                    
                    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                           color=edge_color,  # 线条颜色跟随节点
                           linewidth=width, 
                           alpha=alpha_val,   # 清晰可见
                           zorder=2,
                           solid_capstyle='round')
            
            # --- B. 绘制节点 (使用 node_color_map) ---
            nodes_x = [pos[n][0] for n in G.nodes()]
            nodes_y = [pos[n][1] for n in G.nodes()]
            node_labels = list(G.nodes())
            
            node_fill_colors = [node_color_map.get(n, 'red') for n in node_labels]
            
            # 阴影
            ax.scatter(nodes_x, nodes_y, s=450, c='black', alpha=0.15, linewidths=0, zorder=2.5)
            # 实体
            ax.scatter(nodes_x, nodes_y, s=350, c=node_fill_colors, 
                      edgecolors='white', linewidths=2.5, 
                      alpha=1.0, zorder=3)
            
            # --- C. 标签 ---
            text_halo = [path_effects.withStroke(linewidth=3, foreground="white", alpha=0.9)]
            for n in G.nodes():
                t = ax.text(pos[n][0], pos[n][1], n, 
                       fontsize=10, fontweight='bold', ha='center', va='center', 
                       color='#222222', zorder=4)
                t.set_path_effects(text_halo)
        
        if background_type == 'image':
            ax.invert_yaxis()
            
        ax.axis('off')
        ax.set_title("Spatially-Embedded Interaction Network", fontsize=16, fontweight='bold', pad=20, color='#333333')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"✅ Beautified Light network plot saved: {save_path}")
        
    # =========================================================
    #  绘图功能 3: 空间网络图 (Dark Mode)
    # =========================================================

    def plot_spatial_network_dark(self, interaction_matrix, save_path, 
                                   background_type='celltype', transparency=0.3):
        scale_factor = 1.0
        if background_type == 'image':
            scale_factor = self._get_scale_factor()
        
        display_coords = self.coords * scale_factor
        
        centroids = {}
        unique_clusters = sorted(list(set(interaction_matrix.index).union(set(interaction_matrix.columns))))
        for c in unique_clusters:
            mask = self.adata.obs['target_cluster'].astype(str) == str(c)
            if mask.sum() > 0:
                centroids[str(c)] = np.mean(display_coords[mask], axis=0)
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 12))
        bg_color = '#0a0a0a' # 统一定义背景色
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        if background_type == 'image' and 'spatial' in self.adata.uns:
            self._plot_visium_image_aligned(ax)
        else:
            s_size = 1
            self._plot_celltype_scatter(ax, display_coords, s_size=s_size, is_dark=True)

        G = nx.Graph()
        max_weight = interaction_matrix.values.max() if not interaction_matrix.empty else 1
        threshold = max_weight * 0.05
        
        for src in interaction_matrix.index:
            for tgt in interaction_matrix.columns:
                w = interaction_matrix.loc[src, tgt]
                if w > threshold and str(src) in centroids and str(tgt) in centroids:
                    if G.has_edge(str(src), str(tgt)): 
                        G[str(src)][str(tgt)]['weight'] += w
                    else: 
                        G.add_edge(str(src), str(tgt), weight=w)

        if len(G.nodes()) > 0:
            pos = {n: centroids[n] for n in G.nodes()}
            edges = list(G.edges(data=True))
            weights = [d['weight'] for _, _, d in edges]
            if weights:
                norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
                edge_cmap = cm.get_cmap('plasma')
                edge_colors = [edge_cmap(norm(w)) for w in weights]
                edge_widths = [1.5 + (w/max_weight)*5 for w in weights]
                for (u, v, d), width, color in zip(edges, edge_widths, edge_colors):
                    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                           color=color, linewidth=width*2, alpha=0.3, zorder=2)
                    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                           color=color, linewidth=width, alpha=0.9, zorder=2)
            
            nodes_x = [pos[n][0] for n in G.nodes()]
            nodes_y = [pos[n][1] for n in G.nodes()]
            
            # --- 【修改处 1】 节点绘制：去掉白色描边，改用深色描边或无描边 ---
            ax.scatter(nodes_x, nodes_y, 
                       s=450,           # 稍微调大一点点，让文字更好放下
                       c='#FFFF00',     # 保持黄色节点
                       edgecolors=bg_color, # 将描边设为背景色
                       linewidths=2,    # 增加一点描边厚度，形成“切割”感，使节点更立体
                       alpha=1.0, 
                       zorder=3)

            # --- 【修改处 2】 文字绘制：增加阴影/描边效果，确保极其清晰 ---
            import matplotlib.patheffects as path_effects
            for n in G.nodes():
                txt = ax.text(pos[n][0], pos[n][1], n, 
                             fontsize=10, fontweight='bold',
                             ha='center', va='center', 
                             color='white', # 字体保持白色
                             zorder=4)
                # 给白色字体加一个黑色细边，防止在亮色（如黄色节点）上看不清
                txt.set_path_effects([
                    path_effects.withStroke(linewidth=2, foreground='black')
                ])
        
        self._add_network_legend_enhanced(ax, G, max_weight, cm.get_cmap('plasma'))

        if background_type == 'image':
            ax.invert_yaxis()
            
        ax.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        plt.close()
        plt.style.use('default') 
        logger.info(f"✅ Dark network plot saved: {save_path}")

    # =========================================================
    #  辅助绘图函数
    # =========================================================

    def _plot_visium_image_aligned(self, ax):
        try:
            lib_id = list(self.adata.uns['spatial'].keys())[0]
            img = self.adata.uns['spatial'][lib_id]['images']['hires']
            ax.imshow(img, zorder=0) 
        except Exception as e:
            logger.warning(f"⚠️ Failed to plot Visium image: {e}")

    def _plot_celltype_scatter(self, ax, coords, s_size=50, is_dark=False):
        """
        [重点修改] 绘制清晰的细胞类型背景
        增加了 alpha，移除了 edgecolors 以保证在 s_size 较小时仍可见
        """
        if 'target_cluster' not in self.adata.obs:
            c = '#aaaaaa' if is_dark else 'lightgrey'
            ax.scatter(coords[:, 0], coords[:, 1], c=c, s=s_size, zorder=1)
            return

        unique_clusters = self.adata.obs['target_cluster'].astype(str).unique()
        cmap_name = 'Set3' if is_dark else 'tab20'
        cmap = cm.get_cmap(cmap_name, len(unique_clusters))
        cluster_to_color = {c: cmap(i) for i, c in enumerate(unique_clusters)}
        
        colors = [cluster_to_color.get(str(c)) for c in self.adata.obs['target_cluster']]
        
        # alpha=0.9: 更不透明
        # edgecolors='none': 防止小点的颜色被边框吃掉
        ax.scatter(coords[:, 0], coords[:, 1], 
                 c=colors, s=s_size, 
                 linewidths=0, edgecolors='none', 
                 alpha=0.9, zorder=1) 
                 
    def _add_network_legend_enhanced(self, ax, G, max_weight, edge_cmap):
        from matplotlib.lines import Line2D
        legend_elements = []
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan' if max_weight>0 else 'white', markersize=10, label='Cluster'))
        if max_weight > 0:
            legend_elements.append(Line2D([0], [0], color=edge_cmap(0.5), linewidth=2, label='Interaction'))
        
        if legend_elements:
            l = ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.8)
            for text in l.get_texts():
                text.set_color('white' if plt.rcParams['axes.facecolor']=='#0a0a0a' else 'black')

    # =========================================================
    #  其他图 (弦图、桑基图、气泡图)
    # =========================================================

    def plot_chord_diagram(self, interaction_matrix, save_path):
        try:
            from pycirclize import Circos
        except ImportError:
            logger.error("❌ pycirclize not installed.")
            return

        threshold = interaction_matrix.values.max() * 0.1
        matrix_filtered = interaction_matrix.copy()
        matrix_filtered[matrix_filtered < threshold] = 0
        try:
            circos = Circos.initialize_from_matrix(
                matrix_filtered, space=3, cmap="tab20", 
                label_kws=dict(size=10, color="black", orientation="horizontal"), 
                link_kws=dict(ec="black", lw=0.1, alpha=0.6)
            )
            fig = circos.plotfig()
            plt.title(f"Inter-Cluster Communication Chord Diagram", fontsize=14)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✅ Chord Diagram saved: {save_path}")
        except Exception as e:
            logger.error(f"❌ Chord diagram failed: {e}")

    def plot_sankey_flow(self, interaction_matrix, save_path):
        """
        [修复版] 桑基图 - 纯黑文字增强版
        修复内容：
        1. 删除了 font 中非法的 align 参数
        2. 将所有标签、标题、注释文字改为纯黑色 (black) 并加粗，提高清晰度
        """
        try:
            import plotly.graph_objects as go
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm
        except ImportError:
            logger.error("❌ plotly not installed.")
            return

        labels = sorted(interaction_matrix.index.tolist())
        n_clusters = len(labels)
        label_map = {name: i for i, name in enumerate(labels)}
        
        sources, targets, values = [], [], []
        threshold = interaction_matrix.values.max() * 0.10 
        
        for src in labels:
            for tgt in labels:
                w = interaction_matrix.loc[src, tgt]
                if w > threshold:
                    sources.append(label_map[src])
                    targets.append(label_map[tgt] + n_clusters)
                    values.append(w)

        cmap = cm.get_cmap('tab20', n_clusters)
        cluster_hex_colors = [mcolors.to_hex(cmap(i)) for i in range(n_clusters)]
        sankey_node_colors = cluster_hex_colors + cluster_hex_colors

        def hex_to_rgba(hex_color, alpha=0.4):
            rgb = mcolors.to_rgb(hex_color)
            return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"

        link_colors = [hex_to_rgba(sankey_node_colors[src_idx], alpha=0.6) for src_idx in sources]

        fig = go.Figure(data=[go.Sankey(
            textfont = dict(size=12, color="black", family="Arial Black"), # 【修改1】设置桑基图节点文字为纯黑，并加粗字体
            node = dict(
                pad = 20, thickness = 25,
                line = dict(color = "black", width = 0.5), # 【可选】节点边框也改为黑色增加对比度，不喜欢可改回 "white"
                label = labels + labels,
                color = sankey_node_colors,
                hovertemplate='<b>Cluster: %{label}</b><br>Total Flow: %{value:.2f}<extra></extra>'
            ),
            link = dict(
                source = sources, target = targets, value = values,
                color = link_colors,
                hovertemplate='From: <b>%{source.label}</b><br>To: <b>%{target.label}</b><br>Strength: %{value:.2f}<extra></extra>'
            ),
             arrangement = "snap"
        )])

        fig.update_layout(
            title=dict(
                text="<b>Inter-Cellular Communication Flow</b>",
                x=0.5,
                font=dict(family="Arial", size=16, color="black") # 【修改2】标题改为纯黑且字号调大
            ),
            font=dict(color="black"), # 【修改3】全局字体颜色设为纯黑
            height=700, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            annotations=[
                dict(
                    x=0.0, y=1.06, xref='paper', yref='paper', 
                    text='<b>Source</b>', showarrow=False, 
                    font=dict(size=18, color='black') # 【修改4】Source 标签改为纯黑，字号微调大
                ),
                dict(
                    x=1.0, y=1.06, xref='paper', yref='paper', 
                    text='<b>Target</b>', showarrow=False, 
                    font=dict(size=18, color='black'), # 【修改5】Target 标签改为纯黑
                    xanchor='right'
                )
            ],
            margin=dict(t=100, b=30, l=50, r=50)
        )

        try:
            # 增加 scale 可以提高图片导出时的清晰度
            fig.write_image(save_path, scale=3, width=1000, height=700)
            logger.info(f"✅ Sankey Diagram saved: {save_path}")
        except Exception as e:
            logger.warning(f"⚠️ Sankey export failed: {e}")
            fig.write_html(save_path.replace('.pdf', '.html'))
            
    def plot_lr_dotplot(self, results_json, save_path):
        pair_counts = {}
        for entry in results_json:
            txt = entry.get('llm_analysis', '')
            for pair in entry.get('known_interactions', []):
                if pair.split(' -> ')[0] in txt: 
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
        top_pairs = sorted(pair_counts, key=pair_counts.get, reverse=True)[:15]
        if not top_pairs: return
        genes = list(set([p.split(' -> ')[col] for p in top_pairs for col in [0,1]]))
        try:
            sc.settings.set_figure_params(dpi=150, facecolor='white')
            sc.pl.dotplot(
                self.adata, var_names=genes, groupby='target_cluster', 
                standard_scale='var', cmap='Reds', return_fig=True
            ).savefig(save_path)
            plt.close()
            logger.info(f"✅ DotPlot saved: {save_path}")
        except Exception as e:
            logger.warning(f"❌ DotPlot failed: {e}")
            
            

# ==========================================
# 8. [NEW] 3D 专用可视化引擎 (ThreeDVisualizer)
# ==========================================
# ==========================================
# 8. [NEW] 3D 专用可视化引擎 (ThreeDVisualizer - Enhanced)
# ==========================================
class ThreeDVisualizer:
    def __init__(self, adata, config):
        self.adata = adata
        self.cfg = config
        self.coords = adata.obsm['spatial']
        # 确保是 3D 数据
        if self.coords.shape[1] != 3:
            logger.warning("⚠️ Data is not 3D! ThreeDVisualizer might fail.")
            # 如果是2D数据强行补0
            if self.coords.shape[1] == 2:
                self.coords = np.column_stack([self.coords, np.zeros(self.coords.shape[0])])

    def _setup_dark_axis(self, fig, elev=20, azim=45):
        """辅助函数：设置暗黑风格 3D 坐标轴"""
        ax = fig.add_subplot(111, projection='3d')
        # 背景色设为极深灰/黑
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        
        # 移除坐标轴背景板和网格
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # 移除框线
        ax.axis('off')
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        return ax

    def plot_3d_overview(self, save_path="Figure_1_3D_Overview.pdf"):
        """绘制 3D 组织概览（星云风格）"""
        import matplotlib.pyplot as plt
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12, 12))
        ax = self._setup_dark_axis(fig, elev=25, azim=60)
        
        # 1. 绘制背景微尘 (所有细胞)
        # 使用极小的点和低透明度，制造体积感
        ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                   c='white', s=0.5, alpha=0.1, linewidth=0)

        # 2. 绘制 Cluster (高亮)
        if 'target_cluster' in self.adata.obs:
            clusters = self.adata.obs['target_cluster'].astype(str)
            unique_clusters = sorted(clusters.unique())
            # 使用高对比度的 Neon 色彩
            cmap = plt.cm.get_cmap('rainbow', len(unique_clusters))
            
            for i, cid in enumerate(unique_clusters):
                mask = clusters == cid
                # 绘制两层以产生发光效果
                # 内核
                ax.scatter(self.coords[mask, 0], self.coords[mask, 1], self.coords[mask, 2],
                           label=cid, s=2, alpha=0.8, color=cmap(i), linewidth=0, depthshade=True)
                # 外晕 (可选，如果点太多可能会卡)
                # ax.scatter(self.coords[mask, 0], self.coords[mask, 1], self.coords[mask, 2],
                #            s=10, alpha=0.1, color=cmap(i), linewidth=0)
            
            # 图例美化
            leg = ax.legend(bbox_to_anchor=(1.0, 0.8), loc='center left', 
                            frameon=False, fontsize=10, labelcolor='white')
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
                lh._sizes = [30]
        else:
            ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                       s=1, alpha=0.5, c='cyan')

        plt.title("3D Tissue Architecture", color='white', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor='#0a0a0a')
        plt.style.use('default')
        plt.close()
        logger.info(f"✅ 3D Overview saved: {save_path}")

    def plot_3d_network(self, interaction_matrix, save_path="Figure_2_3D_Network.pdf"):
        """
        在 3D 空间中绘制 Cluster 质心交互网络 (赛博朋克风格)
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if interaction_matrix is None or interaction_matrix.empty:
            return

        # 1. 计算质心
        centroids = {}
        clusters = self.adata.obs['target_cluster'].astype(str)
        unique_cls = sorted(clusters.unique())
        
        for c in unique_cls:
            mask = clusters == c
            if mask.sum() > 0:
                centroids[c] = np.mean(self.coords[mask], axis=0)

        # 2. 建图
        G = nx.Graph()
        max_val = interaction_matrix.values.max()
        threshold = max_val * 0.1
        
        for src in interaction_matrix.index:
            for tgt in interaction_matrix.columns:
                w = interaction_matrix.loc[src, tgt]
                if w > threshold and src in centroids and tgt in centroids:
                    if G.has_edge(src, tgt):
                        G[src][tgt]['weight'] += w
                    else:
                        G.add_edge(src, tgt, weight=w)

        # 3. 绘图
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 14))
        ax = self._setup_dark_axis(fig, elev=30, azim=-60)
        
        # A. 绘制背景轮廓 (极其微弱)
        ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], 
                   c='white', s=0.5, alpha=0.02, linewidth=0)

        # B. 绘制节点 (发光球体)
        cmap = plt.cm.get_cmap('plasma', len(unique_cls))
        color_map = {c: cmap(i) for i, c in enumerate(unique_cls)}
        
        for node in G.nodes():
            x, y, z = centroids[node]
            # 外发光
            ax.scatter(x, y, z, c=[color_map.get(node, 'red')], s=500, alpha=0.3, linewidth=0)
            # 核心
            ax.scatter(x, y, z, c=[color_map.get(node, 'red')], s=100, alpha=1.0, edgecolors='white', linewidth=1)
            # 标签
            ax.text(x, y, z, f"  {node}", fontsize=10, fontweight='bold', color='white', zorder=100)

        # C. 绘制边 (光束)
        edges = list(G.edges(data=True))
        if edges:
            weights = [d['weight'] for _, _, d in edges]
            max_w = max(weights)
            
            for u, v, d in edges:
                w = d['weight']
                p1 = centroids[u]
                p2 = centroids[v]
                
                # 强度决定亮度和粗细
                intensity = (w / max_w)
                lw = 0.5 + intensity * 4
                alpha = 0.2 + intensity * 0.8
                color = color_map.get(u, 'cyan') # 连线颜色跟随源节点
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                        c=color, linewidth=lw, alpha=alpha)

        plt.title("3D Spatially-Embedded Interaction Network", color='white', fontsize=18)
        plt.savefig(save_path, dpi=300, facecolor='#0a0a0a')
        plt.style.use('default')
        plt.close()
        logger.info(f"✅ 3D Network saved: {save_path}")

    def plot_3d_interaction_hotspot(self, results_json, save_path="Figure_3_3D_Hotspot.pdf"):
        """
        绘制 3D 空间互作热点 (Magma Fire Style)
        """
        import matplotlib.pyplot as plt
        from scipy import sparse
        
        # ... (寻找 target_pair 的逻辑保持不变) ...
        target_pair = None
        target_cid = None
        for entry in results_json:
            txt = entry.get('llm_analysis', '')
            if 'Segregated' not in txt and entry.get('known_interactions'):
                for pair in entry['known_interactions']:
                    g1, g2 = pair.split(' -> ')
                    if g1 in txt and g2 in txt:
                        target_pair = (g1, g2)
                        target_cid = entry['cluster_id']
                        break
            if target_pair: break
        
        if not target_pair: return

        ga, gb = target_pair
        
        # 计算强度
        try:
            idx_a = self.adata.var_names.get_loc(ga)
            idx_b = self.adata.var_names.get_loc(gb)
            
            def get_val(idx):
                if sparse.issparse(self.adata.X):
                    return self.adata.X[:, idx].toarray().flatten()
                return self.adata.X[:, idx].flatten()

            va = get_val(idx_a)
            vb = get_val(idx_b)
            
            va = (va - va.min()) / (va.max() - va.min() + 1e-9)
            vb = (vb - vb.min()) / (vb.max() - vb.min() + 1e-9)
            
            score = va * vb
            
            # 绘图
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(12, 12))
            ax = self._setup_dark_axis(fig, elev=20, azim=120)
            
            # 1. 幽灵背景
            ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2], 
                       c='white', s=0.5, alpha=0.03, linewidth=0)
            
            # 2. 火焰热点
            mask = score > 0.05
            if mask.sum() > 0:
                # 使用 magma cmap: 黑->紫->红->黄->白，非常适合暗背景
                p = ax.scatter(self.coords[mask, 0], self.coords[mask, 1], self.coords[mask, 2], 
                               c=score[mask], cmap='magma', s=15, alpha=0.9, linewidth=0, depthshade=False)
                
                cbar = fig.colorbar(p, ax=ax, shrink=0.5, pad=0.1)
                cbar.set_label(f"{ga}-{gb} Interaction Strength", color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            plt.title(f"3D Interaction Hotspot: {ga} -> {gb}", color='white', fontsize=16)
            plt.savefig(save_path, dpi=300, facecolor='#0a0a0a')
            plt.style.use('default')
            plt.close()
            logger.info(f"✅ 3D Hotspot saved: {save_path}")

        except Exception as e:
            logger.error(f"❌ 3D Hotspot failed: {e}")