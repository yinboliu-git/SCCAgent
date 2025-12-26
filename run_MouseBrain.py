import os
import importlib
import logging
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # 用于计算簇间距离

# 引入核心模块
import pipeline_llm 

# 强制重载以确保 pipeline_llm.py 的修改生效
importlib.reload(pipeline_llm)

from pipeline_llm import (
    SpaceConfig, 
    DataManager, 
    SpaceAnalyzer, 
    AgentInterface, 
    AdvancedEvaluator, 
    Visualizer, 
    HallucinationAnalyzer, 
    QualityEvaluator,
    AdvancedVisualizer
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 核心功能函数封装
# ==========================================

def build_global_network_data(adata, final_results, cfg):
    """
    [核心修复版] 构建全局交互网络数据 (CellChat 风格 - 质量作用定律 + 距离衰减)
    
    改进点：
    1. 引入 Law of Mass Action: Score = Avg(L)_src * Avg(R)_tgt
    2. 引入 Distance Decay: 距离越近，权重越高，而非简单的 0/1 截断。
    """
    logger.info("🔄 Calculating Weighted Source-Target interactions (Mass Action + Distance Decay)...")
    
    # 1. 准备数据
    if 'target_cluster' not in adata.obs:
        return None

    adata.obs['target_cluster'] = adata.obs['target_cluster'].astype(str)
    unique_clusters = adata.obs['target_cluster'].unique()
    
    # 2. 预计算所有 Cluster 的基因表达均值 (Pre-compute Mean Expression)
    # 这步至关重要，用于后续计算 L*R 强度
    logger.info("   - Pre-computing cluster gene expression profiles (Mean Expression)...")
    cluster_expr_means = {}
    
    # 为了加速，先转换成 DataFrame 或者直接操作 numpy
    # 注意：这里假设数据已经归一化 (log1p)，如果是 count 需要先归一化
    for cid in unique_clusters:
        mask = adata.obs['target_cluster'] == cid
        if mask.sum() == 0: continue
        
        # 获取该 Cluster 的平均表达量
        # 处理稀疏矩阵
        X_subset = adata.X[mask]
        if hasattr(X_subset, "toarray"):
            mean_expr = np.array(X_subset.mean(axis=0)).flatten()
        else:
            mean_expr = np.array(X_subset.mean(axis=0)).flatten()
            
        # 存入字典，Key为基因名，Value为表达值
        cluster_expr_means[cid] = pd.Series(mean_expr, index=adata.var_names)

    # 3. 准备空间计算参数
    if 'spatial_norm' in adata.obsm:
        coords_all = adata.obsm['spatial_norm']
        dist_thresh = getattr(cfg, 'adaptive_threshold', 0.05)
        logger.info(f"   - Using Normalized Coords (Threshold={dist_thresh:.4f})")
    else:
        coords_all = adata.obsm['spatial']
        dist_thresh = 250.0

    # 4. 构建边列表
    matrix_data = []
    
    for entry in final_results:
        src_cluster = str(entry['cluster_id'])
        txt = entry.get('llm_analysis', '')
        
        # 遍历 LLM 认为有效的互作对
        for pair in entry.get('known_interactions', []):
            g1, g2 = pair.split(' -> ') # g1=Ligand, g2=Receptor
            
            # A. 必须通过 LLM 语义验证
            if (g1 in txt and g2 in txt) and ("Segregated" not in txt):
                
                # 获取 Source 的配体表达量
                if src_cluster not in cluster_expr_means: continue
                val_L = cluster_expr_means[src_cluster].get(g1, 0.0)
                
                # 如果 Source 根本不表达这个配体 (或者极低)，则跳过
                if val_L < 0.01: continue 

                # 获取 Source 坐标
                mask_src = adata.obs['target_cluster'] == src_cluster
                coords_src = coords_all[mask_src]
                
                # 遍历所有 Target Clusters
                for tgt_cluster in unique_clusters:
                    # B. 获取 Target 的受体表达量
                    if tgt_cluster not in cluster_expr_means: continue
                    val_R = cluster_expr_means[tgt_cluster].get(g2, 0.0)
                    
                    # 如果 Target 不表达受体，通讯强度为 0
                    if val_R < 0.01: continue
                    
                    # === 核心改进 1: 质量作用定律 (Mass Action) ===
                    # 基础强度 = Ligand_Expr * Receptor_Expr
                    # 这确保了受体表达量高的 Cluster 获得更高的权重
                    base_strength = val_L * val_R
                    
                    min_dist = float('inf')
                    
                    # C. 空间距离计算
                    if src_cluster == tgt_cluster:
                        # 自分泌：距离视为 0，权重最高
                        min_dist = 0.0
                        dist_factor = 1.0
                    else:
                        mask_tgt = adata.obs['target_cluster'] == tgt_cluster
                        coords_tgt = coords_all[mask_tgt]
                        
                        if len(coords_tgt) > 0:
                            # 计算最近距离
                            dists = cdist(coords_src, coords_tgt)
                            min_dist = dists.min()
                            
                            # === 核心改进 2: 距离衰减 (Soft Threshold) ===
                            if min_dist > dist_thresh:
                                dist_factor = 0.0 # 超过阈值，截断
                            else:
                                # 线性衰减：距离越近 (0)，因子越接近 1；距离接近阈值，因子接近 0
                                dist_factor = 1.0 - (min_dist / dist_thresh)
                                # 或者使用指数衰减 (更平滑): 
                                # dist_factor = np.exp(-min_dist / (dist_thresh * 0.5))
                        else:
                            dist_factor = 0.0

                    # D. 最终评分
                    final_score = base_strength * dist_factor
                    
                    if final_score > 0:
                        matrix_data.append({
                            'source': src_cluster, 
                            'target': tgt_cluster, 
                            'count': 1,              # 保留计数供参考
                            'strength': final_score, # [新增] 真实的物理化学强度
                            'lr_pair': pair
                        })
    
    if not matrix_data:
        return None

    return pd.DataFrame(matrix_data)

# ==========================================
# 主程序入口
# ==========================================

# ==========================================
# 主程序入口 (Modified for GPT-4 Support)
# ==========================================

if __name__ == "__main__":
    # 1. 初始化配置
    data_name = 'squidpy_mouse_brain'
    print("🚀 Initializing SpaceAgent Framework...")
    
    # ===【关键修改点】在此处配置 LLM 模式和密钥 ===
    cfg = SpaceConfig(
        data_dir="./gold_data", 
        result_dir=f"./results_gold/{data_name}/",
        llm_source="api", 
        openai_api_key="sk-XXXXXXXXXXXXXXXXXXXXXXXX",  # <--- 请在此处填入你的真实密钥
        openai_model="gpt-4", 
        gpu_id=0
    )

    # 初始化各个模块
    dm = DataManager(cfg)
    sa = SpaceAnalyzer(cfg)
    agent = AgentInterface(cfg) # 这里会自动读取 cfg.llm_source 来决定加载哪个模型

    # 2. 数据准备
    print("\n📦 Preparing Data...")
    dm.prepare_knowledge_base()
    h5ad_path = dm.load_dataset(source=data_name) 

    # 3. 预处理与拓扑分析
    print("\n🔬 Preprocessing & Analyzing Topology...")
    json_data, adata = dm.preprocess_adata(h5ad_path)
    
    if 'target_cluster' in adata.obs:
        print(f"✅ Using Cluster Column: {adata.obs['target_cluster'].name}")
        print(f"   Clusters: {adata.obs['target_cluster'].unique().tolist()}")
    else:
        print("⚠️ Warning: No cluster column found.")

    # 生成拓扑报告 (注入功能注释)
    json_data = sa.generate_reports(json_data, adata, data_manager=dm)
    
    # 4. LLM 推理 (Agent 会根据 cfg 自动调用 GPT-4)
    print(f"\n🤖 Running LLM Inference (Mode: {cfg.llm_source})...")
    agent.load_model() # 如果是 API 模式，这里会初始化 OpenAI Client
    final_results = agent.run_inference(json_data)
    
    # 备份推理结果
    pd.DataFrame(final_results).to_json(
        os.path.join(cfg.result_dir, "raw_inference_results.json")
    )

    # ... (后续代码保持不变: 幻觉测试、可视化等) ...
    # 5. 幻觉测试 (Figure 4)
    print("\n📉 Running Hallucination Evaluation (Figure 4)...")
    hallucination_checker = HallucinationAnalyzer(sa)
    dist_data = hallucination_checker.collect_distances(adata, final_results)

    Visualizer.plot_hallucination_test(
        dist_data, 
        save_path=os.path.join(cfg.result_dir, "Figure_4_Hallucination.pdf")
    )

    # 6. 定量基准测试 (Figure 5 & Table)
    print("\n⚖️ Running Quantitative Benchmark (Figure 5)...")
    ev = AdvancedEvaluator(adata, cfg)
    df_res = ev.run_benchmark(final_results)

    if df_res is not None:
        Visualizer.plot_benchmark_metrics(
            df_res, 
            save_path=os.path.join(cfg.result_dir, "Figure_5_Metrics.pdf")
        )
        df_res.to_csv(os.path.join(cfg.result_dir, "benchmark_metrics.csv"), index=False)
        
        print("\n🔍 Fairness Check (Total Signal Retention):")
        for m in ["GeneAgent", "SpaceAgent"]:
            sub = df_res[df_res["Model"]==m]
            if not sub.empty:
                print(f"   - {m}: Avg Score={sub['Neighbor_Score'].mean():.4f}")

    # 7. 定性评分 (Table 3) - 也会自动使用 GPT-4 作为裁判
    print("\n👩‍⚖️ Running Qualitative Quality Evaluation (Table 3)...")
    judge = QualityEvaluator(cfg)
    quality_df = judge.evaluate_fairness(final_results)
    if not quality_df.empty:
        quality_df.to_csv(os.path.join(cfg.result_dir, "quality_scores.csv"), index=False)

    # 8. 高级可视化
    print("\n🎨 Generating Publication-Ready Figures...")

    # 8.1 基础图
    if len(final_results) > 0:
        av_base = Visualizer()
        Visualizer.plot_interaction_spatial(
            adata, final_results[0], 
            save_path=os.path.join(cfg.result_dir, "Figure_3A_Example.pdf")
        )
        Visualizer.plot_lr_colocalization_heatmap(
            adata, final_results[0],
            save_path=os.path.join(cfg.result_dir, "Figure_3B_Halo.pdf")
        )

    # 8.2 全貌网络图构建
    results_df = build_global_network_data(adata, final_results, cfg)

    if results_df is not None:
        av = AdvancedVisualizer(adata, cfg)
        
        matrix = av.plot_cluster_interaction_matrix(
            results_df, 
            value_col='strength',
            save_path=os.path.join(cfg.result_dir, "Figure_5_Interaction_Matrix.pdf")
        )

        if matrix is not None:
            print("🌃 Generating Enhanced Dark-Mode Spatial Network...")
            av.plot_spatial_network_dark(
                matrix, 
                save_path=os.path.join(cfg.result_dir, "Figure_1_Spatial_Network_Dark.pdf"),
                background_type='celltype',
                transparency=0.5
            )
            
            print("🌞 Generating Light-Mode Version for Comparison...")
            av.plot_spatial_network_light(
                matrix, 
                save_path=os.path.join(cfg.result_dir, "Figure_1_Spatial_Network_Light.pdf"),
                background_type='celltype'
            )
            
            # 如果需要带 H&E 背景图的版本 (仅 Visium 数据有效)
            av.plot_spatial_network_light(
                matrix, 
                save_path=os.path.join(cfg.result_dir, "Figure_1_Spatial_Network_Light_imgs.pdf"),
                background_type='image'
            )
            
            print("🎻 Generating PyCirclize Chord Diagram...")
            av.plot_chord_diagram(
                matrix, 
                save_path=os.path.join(cfg.result_dir, "Figure_6_Chord_Circlize.pdf")
            )

            print("🌊 Generating Sankey Diagram...")
            av.plot_sankey_flow(
                matrix, 
                save_path=os.path.join(cfg.result_dir, "Figure_8_Sankey.pdf")
            )
            
    print("🫧 Generating DotPlot...")
    av = AdvancedVisualizer(adata, cfg)
    av.plot_lr_dotplot(
        final_results, 
        save_path=os.path.join(cfg.result_dir, "Figure_7_LR_DotPlot.pdf")
    )

    print("\n✅ All experiments completed! Please check the './results_gold' folder.")