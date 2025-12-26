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
# 1. 配置管理
# ==========================================
@dataclass
@dataclass
# ==========================================
# 1. 配置管理 (Revised)
# ==========================================
@dataclass
class SpaceConfig:
    data_dir: str = "./gold_data"
    result_dir: str = "./results_gold"
    
    # 知识库路径
    db_path: str = "omnipath_intercell.csv"
    kegg_path: str = "KEGG_2021_Human.gmt"
    go_bp_path: str = "GO_Biological_Process_2023.gmt"
    reactome_path: str = "Reactome_2022.gmt"
    
    # Data Processing Parameters
    min_counts: int = 50
    min_genes: int = 200
    n_marker_genes: int = 1000
    blacklist_prefixes: Tuple[str] = ("MT-", "RPS", "RPL", "HB", "MALAT1")
    spatial_dist_threshold: float = 250.0
    
    # --- LLM Settings (Modified) ---
    # mode options: 'local' (for Llama) or 'api' (for GPT-4)
    llm_source: str = "api" 
    
    # Local Model Settings
    model_id: str = None
    gpu_id: int = 0
    
    # OpenAI / GPT-4 Settings
    openai_api_key: str = "sk-XXX"  # <--- REPLACE WITH YOUR KEY
    openai_model: str = "gpt-4"     # Or 
    openai_base_url: str = "https://api.openai.com/v1" # Optional: For proxies
    
    max_new_tokens: int = 512

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

# ==========================================
# 2. 数据管理
# ==========================================
class DataManager:
    def __init__(self, config: SpaceConfig):
        self.cfg = config
        self.lr_db = {}
        self.all_lr_genes = set()
        
        # [新增] 功能注释缓存 (Gene -> List of Functions)
        self.func_db = {} 

    def _download_gmt(self, url, save_path):
        """辅助函数：下载 GMT 格式数据库"""
        if not os.path.exists(save_path):
            print(f"⬇️ Downloading DB to {save_path}...")
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(r.text)
            except Exception as e:
                logger.warning(f"⚠️ Download failed: {e}")

    def prepare_knowledge_base(self):
        # 1. OmniPath (保持不变)
        if not os.path.exists(self.cfg.db_path):
            print("⬇️ Downloading OmniPath...")
            try:
                import omnipath as op
                db = op.interactions.import_intercell_network(
                    transmitter_params={"categories": "ligand"},
                    receiver_params={"categories": "receptor"}
                )
                db.to_csv(self.cfg.db_path, index=False)
            except Exception: pass
        
        # 加载 OmniPath
        try:
            df = pd.read_csv(self.cfg.db_path, low_memory=False)
            # ... (保留原有的 OmniPath 解析代码) ...
            cols = df.columns
            src_col = next((c for c in ['source_genesymbol', 'genesymbol_intercell_source'] if c in cols), None)
            tgt_col = next((c for c in ['target_genesymbol', 'genesymbol_intercell_target'] if c in cols), None)
            if src_col and tgt_col:
                for _, row in df.iterrows():
                    s, t = str(row[src_col]).upper().strip(), str(row[tgt_col]).upper().strip()
                    if s and t and s!=t:
                        self.all_lr_genes.add(s); self.all_lr_genes.add(t)
                        self.lr_db[(s, t)] = "LR"
        except: pass

        # 2. [新增] 下载并加载功能数据库 (GO & Reactome)
        # 使用 Enrichr 的库源
        self._download_gmt(
            "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023",
            self.cfg.go_bp_path
        )
        self._download_gmt(
            "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Reactome_2022",
            self.cfg.reactome_path
        )

        # 3. [新增] 解析功能库到内存 (构建 Gene -> Function 映射)
        self._load_func_db(self.cfg.go_bp_path, source="GO")
        self._load_func_db(self.cfg.reactome_path, source="Reactome")
        
        print(f"✅ Knowledge Base Ready: OmniPath + {len(self.func_db)} functional annotations.")

    def _load_func_db(self, path, source="DB"):
        """读取 GMT 文件并建立反向索引: Gene -> [Function1, Function2]"""
        if not os.path.exists(path): return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue
                    term_name = parts[0]  # 功能名称
                    genes = parts[2:]     # 基因列表
                    
                    for g in genes:
                        g_upper = g.upper()
                        if g_upper not in self.func_db:
                            self.func_db[g_upper] = []
                        # 限制每个基因最多存 5 条最具体的功能，防止 Prompt 爆炸
                        if len(self.func_db[g_upper]) < 5:
                            # 简单清洗：去掉太长的描述
                            if len(term_name) < 80:
                                self.func_db[g_upper].append(f"{term_name} ({source})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {path}: {e}")

    def load_dataset(self, source: str = "squidpy_mouse_brain") -> str:
        """
        加载数据集，支持自定义数据
        """
        dataset_map = {
            # 1. 基准: Mouse Brain (Visium)
            "squidpy_mouse_brain": {
                "filename": "visium_mouse_brain_hne.h5ad",
                "desc": "Visium H&E Mouse Brain",
                "type": "squidpy",
                "func": sq.datasets.visium_hne_adata
            },
            # 2. 高精: Mouse Embryo (seqFISH)
            "squidpy_seqfish": {
                "filename": "seqfish_mouse_embryo.h5ad",
                "desc": "seqFISH Mouse Embryo",
                "type": "squidpy",
                "func": sq.datasets.seqfish
            },
            # 4. 【新增】用户自定义: Human Embryo 3D
            "custom_human_embryo": {
                "filename": "human_embryo_3D.h5ad", # 目标文件名
                "desc": "Custom Human Embryo (3D Aligned)",
                "type": "local_file",
                "path": "./gold_data/human_embryo_3D.h5ad" # 你的原始文件路径
            }
        }

        if source not in dataset_map:
            logger.error(f"Unknown data source: {source}")
            return None

        meta = dataset_map[source]
        
        # 目标保存路径 (统一管理在 result_dir 或 data_dir)
        # 注意：对于 local_file，我们直接使用原始路径，或者由用户指定
        if meta["type"] == "local_file":
            save_path = meta["path"]
        else:
            save_path = os.path.join(self.cfg.data_dir, meta["filename"])
        
        # 1. 检查文件是否存在
        if os.path.exists(save_path):
            logger.info(f"✅ Found existing data ({meta['desc']}): {save_path}")
            # 如果是自定义文件，我们做一次简单的读取检查，确保它是有效的 h5ad
            if meta["type"] == "local_file":
                return save_path 
            # 对于下载类数据，继续后续逻辑...
            return save_path
            
        # 2. 下载 (针对非本地文件)
        logger.info(f"⬇️ Downloading {meta['desc']}...")
        try:
            if meta["type"] == "squidpy":
                adata = meta["func"]()
            elif meta["type"] == "squidpy_custom":
                adata = sq.datasets.visium(sample_id=meta["sample_id"], include_hires_tiff=True)
            elif meta["type"] == "local_file":
                logger.error(f"❌ Custom file not found at: {save_path}")
                return None

            # 3. 统一预处理 (同前)
            adata.var_names = [str(g).upper() for g in adata.var_names]
            adata.var_names_make_unique()
            
            # 兼容处理... (省略，与之前一致)
            if source == "squidpy_seqfish" and 'celltype_mapped_refined' in adata.obs.columns:
                adata.obs['cluster'] = adata.obs['celltype_mapped_refined']

            adata.write(save_path)
            logger.info(f"✅ Download & Formatting complete: {adata.shape}")
            return save_path
        except Exception as e:
            logger.error(f"❌ Load failed: {e}")
            return None

    def preprocess_adata(self, adata_path: str):
        # ... (保持原有的 preprocess_adata 代码完全不变) ...
        # 请务必保留之前那版包含 "target_col识别逻辑" 和 "Marker提取" 的代码
        logger.info(f"🔬 Preprocessing {adata_path}...")
        adata = sc.read_h5ad(adata_path)
        
        adata.var_names = [g.upper() for g in adata.var_names]
        adata.var_names_make_unique()
        if adata.raw: del adata.raw
        
        # 简单质控
        sc.pp.filter_cells(adata, min_counts=self.cfg.min_counts)
        # 检查是否需要归一化
        max_val = adata.X.data.max() if sparse.issparse(adata.X) else adata.X.max()
        if max_val > 20:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        # 智能识别聚类标签
        target_col = None
        # 增加 'annotation', 'celltype' 等常见列名
        for col in ['cluster', 'clusters', 'leiden', 'CellType', 'cell_type', 'annotation', 'celltype_mapped_refined']:
            if col in adata.obs.columns:
                target_col = col
                break
        
        if not target_col:
            logger.warning("⚠️ No annotation found, running Leiden...")
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata)
            target_col = 'leiden'
        
        adata.obs['target_cluster'] = adata.obs[target_col].astype('category')
        logger.info(f"   Using annotation: '{target_col}'")
        
        # 检查空间坐标 (关键！针对无图像数据)
        if 'spatial' not in adata.obsm:
            # 尝试寻找常见的坐标列
            if 'X_spatial' in adata.obsm:
                adata.obsm['spatial'] = adata.obsm['X_spatial']
            elif 'spatial' in adata.uns:
                pass # Visium标准格式
            else:
                # 最后的尝试：看obs里有没有 x, y 或 spatial_1, spatial_2
                candidates = [['x', 'y'], ['spatial_1', 'spatial_2'], ['x_centroid', 'y_centroid']]
                for c1, c2 in candidates:
                    if c1 in adata.obs.columns and c2 in adata.obs.columns:
                        adata.obsm['spatial'] = adata.obs[[c1, c2]].values
                        logger.info(f"   ✅ Constructed spatial coords from obs['{c1}', '{c2}']")
                        break
        
        # === [新增] 空间坐标归一化与自适应阈值计算 ===
        logger.info("📏 Normalizing Spatial Coordinates & Auto-detecting Threshold...")
        
        # 1. 获取原始坐标
        raw_coords = adata.obsm['spatial']
        
        # 2. 归一化到 [0, 1] 区间
        min_vals = raw_coords.min(axis=0)
        max_vals = raw_coords.max(axis=0)
        scale = max_vals - min_vals
        # 防止除以0
        scale[scale == 0] = 1.0 
        
        norm_coords = (raw_coords - min_vals) / scale
        adata.obsm['spatial_norm'] = norm_coords # 保存归一化坐标供后续使用
        
        # 3. 计算自适应阈值 (基于最近邻距离)
        # 逻辑：计算所有点的平均最近邻距离 (Average Nearest Neighbor Distance)
        # 设定阈值为 NND 的倍数 (例如 3-5 倍，代表 3-5 个细胞直径/Spot间距)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(norm_coords)
        distances, _ = nbrs.kneighbors(norm_coords)
        
        # distances[:, 1] 是到最近邻居的距离 (第0个是自己)
        avg_nnd = np.mean(distances[:, 1])
        
        # 动态设定阈值：例如 4 倍平均间距 (即允许跨越 3-4 个细胞/Spot 进行通讯)
        adaptive_threshold = avg_nnd * 4.0
        
        # 将参数注入到 config 中供后续使用
        self.cfg.adaptive_threshold = adaptive_threshold
        self.cfg.unit_scale = "normalized_units"
        
        logger.info(f"   ✅ Auto-Threshold: {adaptive_threshold:.4f} (based on 4x Avg NND: {avg_nnd:.4f})")
        logger.info(f"   ✅ Coordinates normalized to [0, 1].")
        
        # 提取 Marker (保持不变)
        logger.info(f"   Identifying Markers (Top {self.cfg.n_marker_genes})...")
        sc.tl.rank_genes_groups(adata, groupby='target_cluster', method='t-test', use_raw=False, n_genes=self.cfg.n_marker_genes)
        
        extracted_data = []
        for cid in adata.obs['target_cluster'].unique():
            try:
                df_markers = sc.get.rank_genes_groups_df(adata, group=cid)
                markers = df_markers['names'].tolist()
                clean_markers = [g for g in markers if not any(g.startswith(p) for p in self.cfg.blacklist_prefixes)]
                
                search_space = clean_markers[:200] 
                pairs = []
                for g1 in search_space:
                    for g2 in search_space:
                        if (g1, g2) in self.lr_db:
                            pairs.append(f"{g1} -> {g2}")
                
                if pairs:
                    pairs = list(set(pairs))[:50]
                    coords = adata.obsm['spatial'][adata.obs['target_cluster'] == cid]
                    extracted_data.append({
                        "cluster_id": str(cid),
                        "gene_list": clean_markers[:50],
                        "known_interactions": pairs,
                        "coordinates": coords.tolist()
                    })
            except Exception as e:
                pass

        logger.info(f"📊 Extracted {len(extracted_data)} clusters.")
        return extracted_data, adata

# ==========================================
# 3. 空间分析器 (修正截断问题)
# ==========================================
class SpaceAnalyzer:
    def __init__(self, config):
        self.cfg = config

    def _get_morphology_metrics(self, coords: np.ndarray, tissue_centroid: np.ndarray) -> str:
        """
        计算形态学特征 (使用归一化坐标计算，以保证尺度统一)
        """
        if len(coords) < 4: return "Too few cells to determine morphology."
        
        try:
            # 1. 计算 MNND (Mean Nearest Neighbor Distance)
            # 使用 k=2，因为第1个最近邻是自己
            nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
            distances, _ = nbrs.kneighbors(coords)
            mnnd = np.mean(distances[:, 1]) 
            
            # 2. 计算凸包与密度
            hull = scipy.spatial.ConvexHull(coords)
            volume = hull.volume
            # 密度 = 细胞数 / 归一化体积
            density = len(coords) / volume if volume > 0 else 0
            
            # 3. Global Localization (相对于组织几何中心的偏移)
            cluster_centroid = np.mean(coords, axis=0)
            dist_to_center = np.linalg.norm(cluster_centroid - tissue_centroid)
            
            # 4. 生成语义描述 (基于归一化后的经验阈值)
            # 注意：这里的阈值可能需要根据归一化后的分布微调，这里使用相对通用的判断
            cohesion_desc = "highly cohesive" if density > 100 else "dispersed" 
            loc_desc = "peripheral" if dist_to_center > 0.4 else "central" # 归一化坐标范围 0-1，0.4 算边缘
            
            report = (
                f"   - Morphology: The cluster is {cohesion_desc} (Density: {density:.2f}, MNND: {mnnd:.4f}).\n"
                f"   - Localization: Located in the {loc_desc} region (Dist to center: {dist_to_center:.2f})."
            )
            return report
        except Exception as e:
            return f"Morphology calculation failed: {str(e)}"

    def verify_interaction(self, adata, ga, gb, cid):
        """
        验证配受体相互作用的空间距离。
        
        [关键修订]:
        1. 优先使用 'spatial_norm' (归一化坐标)。
        2. Source 限制在当前 Cluster，但 Target 搜索全组织 (解决 Figure 5 非对角线为0的问题)。
        """
        # 1. 确定 Source 候选池（当前 Cluster 的索引）
        source_cluster_indices = np.where(adata.obs['target_cluster'].astype(str) == str(cid))[0]
        
        if len(source_cluster_indices) == 0: 
            return "Silent"

        try:
            # 辅助函数：安全获取表达量
            def get_gene_expr(gene):
                try:
                    idx = adata.var_names.get_loc(gene)
                    col = adata.X[:, idx]
                    if sparse.issparse(col):
                        return col.toarray().flatten()
                    return col.flatten()
                except KeyError:
                    return None

            ea_full = get_gene_expr(ga) # 配体全量
            eb_full = get_gene_expr(gb) # 受体全量
            
            if ea_full is None or eb_full is None: 
                return "Error"
            
            # 2. 构建 Mask
            # Source Mask: 必须属于当前 Cluster 且表达配体
            mask_source = np.zeros(adata.n_obs, dtype=bool)
            mask_source[source_cluster_indices] = True
            mask_source = mask_source & (ea_full > 0)
            
            # Target Mask: 【核心修复】可以是组织中任何表达受体的细胞 (Global Search)
            mask_target = eb_full > 0
            
            # 3. 快速检查
            if mask_source.sum() == 0 or mask_target.sum() == 0:
                return "Silent"
            
            # 检查共定位 (同一个细胞既表达A又表达B)
            if (mask_source & mask_target).sum() > 5:
                return 0.0 
            
            # 4. 准备坐标 (优先使用归一化坐标)
            if 'spatial_norm' in adata.obsm:
                coords = adata.obsm['spatial_norm']
            else:
                # 回退机制：如果 DataManager 没做归一化，就用原始坐标
                coords = adata.obsm['spatial']

            coords_source = coords[mask_source]
            coords_target = coords[mask_target]
            
            # 5. 计算最小距离 (Paracrine Search)
            # cdist 计算 Source 集合到 Target 集合的两两距离矩阵
            dists = scipy.spatial.distance.cdist(coords_source, coords_target)
            
            # 取全局最小值：代表从该 Cluster 发出的信号，到达最近受体的距离
            min_dist = np.min(dists)
            
            return min_dist

        except Exception as e:
            return "Error"

    def generate_reports(self, json_data, adata, data_manager=None):
        """
        生成包含拓扑特征、空间验证和信号强度的综合报告
        [修订版]: 增加了基于 L*R 表达量的信号强度 (Signal Strength) 语义描述
        """
        logger.info("🧠 Generating Spatial Reports with Knowledge & Intensity Injection...")
        
        # 1. 准备全局坐标参数
        if 'spatial_norm' in adata.obsm:
            tissue_coords = adata.obsm['spatial_norm']
        else:
            tissue_coords = adata.obsm['spatial']
        tissue_centroid = np.mean(tissue_coords, axis=0)
        
        # 2. 获取自适应距离阈值
        tau = getattr(self.cfg, 'adaptive_threshold', 0.05)
        
        # 3. 预计算全局受体表达均值 (用于评估受体可用性)
        #    为了加速，我们可以在这里做个简单的缓存或者直接在循环里取
        #    考虑到循环次数不多，直接取也行。
        
        for entry in json_data:
            lines = []
            cid = entry['cluster_id']
            
            # --- Stream A: 拓扑形态学 ---
            mask = adata.obs['target_cluster'].astype(str) == str(cid)
            coords = tissue_coords[mask]
            morphology_report = self._get_morphology_metrics(coords, tissue_centroid)
            lines.append(f"1. Topological Context:\n{morphology_report}")
            
            # --- Stream C: 空间尺度信息 ---
            scale_info = (
                f"2. Spatial Scale Info:\n"
                f"   - Coordinate Space: Normalized [0, 1].\n"
                f"   - Interaction Threshold (Tau): {tau:.4f} units.\n"
                f"   - Logic: Interactions with dist < {tau:.4f} are verified. "
                f"Signal Strength is based on mass action (Ligand * Receptor)."
            )
            lines.append(scale_info)
            
            # --- Stream B: 互作验证与强度评估 ---
            inter_lines = []
            valid_genes = set() 
            
            # 锁定当前 Cluster 的细胞索引 (用于计算 Ligand 局部表达)
            cluster_cells = adata[mask]
            
            for pair in entry['known_interactions'][:40]: 
                ga, gb = pair.split(' -> ')
                
                # 1. 距离验证
                res = self.verify_interaction(adata, ga, gb, cid)
                
                status = "Unknown"
                dist_val = float('inf')
                
                if isinstance(res, str): 
                    status = res
                else:
                    dist_val = res
                    if dist_val < 1e-5: status = "Co-localized (Autocrine)"
                    elif dist_val < tau: status = f"Physically Proximal (Dist: {dist_val:.4f} < Tau)"
                    else: status = f"Spatially Segregated (Dist: {dist_val:.4f} > Tau)"
                
                # 2. 强度计算 (仅针对有效互作)
                intensity_desc = ""
                if "Proximal" in status or "Co-localized" in status:
                    try:
                        # 核心逻辑：强度 = 局部配体表达 * 全局受体表达
                        # (反映该 Cluster 发出信号的能力 x 组织接收信号的潜力)
                        
                        # 获取 Ligand 在当前 Cluster 的均值
                        if sparse.issparse(cluster_cells.X):
                            idx_a = adata.var_names.get_loc(ga)
                            val_L = cluster_cells.X[:, idx_a].mean()
                        else:
                            val_L = cluster_cells[:, ga].X.mean()
                            
                        # 获取 Receptor 在全组织的均值 (或潜在 Target 区域，简化为全组织)
                        if sparse.issparse(adata.X):
                            idx_b = adata.var_names.get_loc(gb)
                            val_R = adata.X[:, idx_b].mean()
                        else:
                            val_R = adata[:, gb].X.mean()
                        
                        # 计算乘积 (Mass Action Proxy)
                        strength_score = val_L * val_R
                        
                        # 语义化映射 (假设数据经过 log1p 处理，值通常在 0-5 之间)
                        # 阈值根据经验设定，可微调
                        if strength_score > 1.0: i_tag = "**Very High**"
                        elif strength_score > 0.5: i_tag = "High"
                        elif strength_score > 0.1: i_tag = "Moderate"
                        else: i_tag = "Low"
                        
                        intensity_desc = f" | Signal Strength: {i_tag}"
                        
                        valid_genes.add(ga)
                        valid_genes.add(gb)
                        
                    except Exception as e:
                        # 容错：如果计算失败，不报错，只留空
                        pass
                    
                inter_lines.append(f"- {ga} -> {gb}: {status}{intensity_desc}")
            
            lines.append("3. Interaction Verification Log:\n" + ("\n".join(inter_lines) if inter_lines else "None"))
            
            # --- Stream D: 外部数据库知识注入 ---
            if data_manager and hasattr(data_manager, 'func_db'):
                kb_lines = []
                for g in list(valid_genes)[:15]:
                    if g in data_manager.func_db:
                        funcs = "; ".join(data_manager.func_db[g][:2]) 
                        kb_lines.append(f"   - {g}: {funcs}")
                
                if kb_lines:
                    lines.append(f"4. Functional Annotations (Background Knowledge):\n" + "\n".join(kb_lines))
                else:
                    lines.append("4. Functional Annotations: None available for valid targets.")
            
            # 合并 Prompt
            entry['spatial_context'] = f"Cluster {cid} Analysis Report:\n" + "\n\n".join(lines)
            
        return json_data

# ==========================================
# 4. Agent 接口
# ==========================================
# ==========================================
# 4. Agent 接口 (Revised with GPT-4 Support)
# ==========================================
class AgentInterface:
    def __init__(self, config: SpaceConfig):
        self.cfg = config
        self.pipe = None
        self.client = None

    def load_model(self):
        """Initializes either the Local LLM or the OpenAI API Client"""
        if self.cfg.llm_source == "api":
            logger.info(f"🚀 Initializing OpenAI API Client ({self.cfg.openai_model})...")
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.cfg.openai_api_key,
                    base_url=self.cfg.openai_base_url
                )
                logger.info("✅ OpenAI Client connected.")
            except ImportError:
                logger.error("❌ 'openai' library not installed. Please run: pip install openai")
            except Exception as e:
                logger.error(f"❌ OpenAI init failed: {e}")
        
        else:
            # Local Mode (GPT, etc.)
            logger.info(f"🚀 Loading Local LLM: {self.cfg.model_id}...")
            try:
                self.pipe = pipeline(
                    "text-generation", 
                    model=self.cfg.model_id, 
                    device=self.cfg.gpu_id, 
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                )
            except Exception as e:
                logger.error(f"❌ Local model load failed: {e}")

    def _call_gpt4(self, messages: List[Dict]) -> str:
        """Helper to call GPT-4 API"""
        if not self.client: return ""
        try:
            response = self.client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=messages,
                temperature=0.2, # Low temp for scientific rigor
                max_tokens=self.cfg.max_new_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"⚠️ GPT-4 API Error: {e}")
            return ""

    def run_inference(self, json_data):
        if (self.cfg.llm_source == "local" and not self.pipe) and \
           (self.cfg.llm_source == "api" and not self.client):
            logger.error("❌ No valid model/client loaded.")
            return json_data

        logger.info(f"🤖 SpaceAgent Reasoning (Source: {self.cfg.llm_source})...")
        results = []
        
        for i, entry in enumerate(json_data):
            # Construct Prompt
            prompt_messages = [
                {"role": "system", "content": "You are a spatial biologist. Reject interactions marked 'Segregated' or 'Silent'. Accept 'Proximal' or 'Co-localized'. Focus on biological plausibility based on the gene functions provided."},
                {"role": "user", "content": f"Genes: {entry.get('gene_list', [])[:20]}\nContext: {entry.get('spatial_context', '')}\n\nAnalyze valid interactions:"}
            ]
            
            output_text = ""
            
            # Branch Logic: API vs Local
            if self.cfg.llm_source == "api":
                output_text = self._call_gpt4(prompt_messages)
            else:
                # Local Pipeline
                try:
                    out = self.pipe(prompt_messages, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
                    output_text = out[0]["generated_text"][-1]["content"]
                except Exception as e:
                    logger.warning(f"⚠️ Local Inference Error: {e}")

            if output_text:
                entry['llm_analysis'] = output_text
                results.append(entry)
                
            # Optional: Print progress for long API runs
            if (i + 1) % 5 == 0:
                logger.info(f"   Processed {i + 1}/{len(json_data)} clusters...")

        return results