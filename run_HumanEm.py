import os
import importlib
import logging
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 

# Import core modules
import pipeline_llm 

# Force reload to ensure modifications in pipeline_llm.py take effect
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
    AdvancedVisualizer,
    ThreeDVisualizer  # [NEW] Import 3D plotting class
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Core Function Encapsulation
# ==========================================

def build_global_network_data(adata, final_results, cfg):
    """
    Build global interaction network data (Compatible with both 2D and 3D).
    """
    logger.info("🔄 Calculating Weighted Source-Target interactions...")
    
    if 'target_cluster' not in adata.obs:
        return None

    adata.obs['target_cluster'] = adata.obs['target_cluster'].astype(str)
    unique_clusters = adata.obs['target_cluster'].unique()
    
    # Pre-compute mean expression
    cluster_expr_means = {}
    for cid in unique_clusters:
        mask = adata.obs['target_cluster'] == cid
        if mask.sum() == 0: continue
        X_subset = adata.X[mask]
        if hasattr(X_subset, "toarray"):
            mean_expr = np.array(X_subset.mean(axis=0)).flatten()
        else:
            mean_expr = np.array(X_subset.mean(axis=0)).flatten()
        cluster_expr_means[cid] = pd.Series(mean_expr, index=adata.var_names)

    # Coordinate preparation (Auto-detect 3D)
    if 'spatial_norm' in adata.obsm:
        coords_all = adata.obsm['spatial_norm']
        dist_thresh = getattr(cfg, 'adaptive_threshold', 0.05)
    else:
        coords_all = adata.obsm['spatial']
        dist_thresh = 250.0

    logger.info(f"   - Network Builder: Using {coords_all.shape[1]}D coordinates.")

    matrix_data = []
    
    for entry in final_results:
        src_cluster = str(entry['cluster_id'])
        txt = entry.get('llm_analysis', '')
        
        for pair in entry.get('known_interactions', []):
            g1, g2 = pair.split(' -> ')
            
            if (g1 in txt and g2 in txt) and ("Segregated" not in txt):
                if src_cluster not in cluster_expr_means: continue
                val_L = cluster_expr_means[src_cluster].get(g1, 0.0)
                if val_L < 0.01: continue 

                mask_src = adata.obs['target_cluster'] == src_cluster
                coords_src = coords_all[mask_src]
                
                for tgt_cluster in unique_clusters:
                    if tgt_cluster not in cluster_expr_means: continue
                    val_R = cluster_expr_means[tgt_cluster].get(g2, 0.0)
                    if val_R < 0.01: continue
                    
                    base_strength = val_L * val_R
                    min_dist = float('inf')
                    
                    if src_cluster == tgt_cluster:
                        min_dist = 0.0
                        dist_factor = 1.0
                    else:
                        mask_tgt = adata.obs['target_cluster'] == tgt_cluster
                        coords_tgt = coords_all[mask_tgt]
                        
                        if len(coords_tgt) > 0:
                            # cdist automatically supports 3D (x,y,z) distance calculation
                            dists = cdist(coords_src, coords_tgt)
                            min_dist = dists.min()
                            
                            if min_dist > dist_thresh:
                                dist_factor = 0.0
                            else:
                                dist_factor = 1.0 - (min_dist / dist_thresh)
                        else:
                            dist_factor = 0.0

                    final_score = base_strength * dist_factor
                    
                    if final_score > 0:
                        matrix_data.append({
                            'source': src_cluster, 
                            'target': tgt_cluster, 
                            'count': 1,
                            'strength': final_score,
                            'lr_pair': pair
                        })
    
    if not matrix_data:
        return None

    return pd.DataFrame(matrix_data)

# ==========================================
# Main Program Entry Point
# ==========================================

if __name__ == "__main__":
    # 1. Initialize Configuration
    data_name = 'custom_human_embryo'  
    print(f"🚀 Initializing SpaceAgent Framework for {data_name}...")
    
    # === [Configuration] Set to use GPT-4 via API ===
    cfg = SpaceConfig(
        data_dir="./gold_data", 
        result_dir=f"./results_gold/{data_name}/",
        
        # --- LLM Mode ---
        llm_source="api", 
        
        # --- API Keys ---
        openai_api_key="sk-XXXXXXXXXXXXXXXXXXXXXXXX",  # <--- REPLACE WITH YOUR REAL KEY
        openai_model="gpt-4", 
        
        # --- Local Fallback ---
        gpu_id=0
    )

    dm = DataManager(cfg)
    sa = SpaceAnalyzer(cfg)
    agent = AgentInterface(cfg)

    # 2. Data Preparation
    print("\n📦 Preparing Data...")
    dm.prepare_knowledge_base()
    h5ad_path = dm.load_dataset(source=data_name) 
    
    if not h5ad_path:
        logger.error("❌ Data path is invalid. Exiting.")
        exit()

    # 3. Preprocessing & Topology Analysis
    print("\n🔬 Preprocessing & Analyzing Topology...")
    json_data, adata = dm.preprocess_adata(h5ad_path)
    
    # 3D Detection
    is_3d_data = False
    if adata.obsm['spatial'].shape[1] == 3:
        is_3d_data = True
        logger.info("🌌 DETECTED 3D DATASET (X, Y, Z). Switching to 3D Visualization Mode.")
    else:
        logger.info("🗺️ DETECTED 2D DATASET. Using Standard Visualization Mode.")

    if 'target_cluster' in adata.obs:
        print(f"✅ Using Cluster Column: {adata.obs['target_cluster'].name}")
    else:
        print("⚠️ Warning: No cluster column found.")

    json_data = sa.generate_reports(json_data, adata, data_manager=dm)

    # 4. LLM Inference
    print(f"\n🤖 Running LLM Inference (Mode: {cfg.llm_source})...")
    agent.load_model()
    final_results = agent.run_inference(json_data)
    
    pd.DataFrame(final_results).to_json(
        os.path.join(cfg.result_dir, "raw_inference_results.json")
    )

    # 5. Hallucination Test
    print("\n📉 Running Hallucination Evaluation...")
    hallucination_checker = HallucinationAnalyzer(sa)
    dist_data = hallucination_checker.collect_distances(adata, final_results)
    Visualizer.plot_hallucination_test(dist_data, save_path=os.path.join(cfg.result_dir, "Figure_4_Hallucination.pdf"))

    # 6. Quantitative Benchmark
    print("\n⚖️ Running Quantitative Benchmark...")
    ev = AdvancedEvaluator(adata, cfg)
    df_res = ev.run_benchmark(final_results)
    if df_res is not None:
        Visualizer.plot_benchmark_metrics(df_res, save_path=os.path.join(cfg.result_dir, "Figure_5_Metrics.pdf"))
        df_res.to_csv(os.path.join(cfg.result_dir, "benchmark_metrics.csv"), index=False)

    # 7. Qualitative Evaluation
    print("\n👩‍⚖️ Running Qualitative Quality Evaluation...")
    judge = QualityEvaluator(cfg)
    quality_df = judge.evaluate_fairness(final_results)
    if not quality_df.empty:
        quality_df.to_csv(os.path.join(cfg.result_dir, "quality_scores.csv"), index=False)

    # ==========================================
    # 8. Visualization Branching (2D vs 3D)
    # ==========================================
    print("\n🎨 Generating Figures...")
    
    # Calculate Interaction Matrix (build_global_network_data uses cdist, supports 3D natively)
    results_df = build_global_network_data(adata, final_results, cfg)
    
    if is_3d_data:
        # =================================
        # 3D Specific Plotting Pipeline
        # =================================
        print("🌌 Entering 3D Visualization Pipeline...")
        t3d = ThreeDVisualizer(adata, cfg)
        
        # 1. 3D Tissue Overview (Beautified)
        t3d.plot_3d_overview(save_path=os.path.join(cfg.result_dir, "Figure_1_3D_Overview.pdf"))
        
        if results_df is not None:
            # 2. 3D Spatial Interaction Network
            # Convert to matrix first for plot_3d_network
            matrix_for_3d = results_df.groupby(['source', 'target'])['strength'].sum().unstack(fill_value=0)
            t3d.plot_3d_network(matrix_for_3d, save_path=os.path.join(cfg.result_dir, "Figure_2_3D_Network.pdf"))
            
            # === [Core Modification] Draw Sankey and Chord even in 3D mode (using AdvancedVisualizer) ===
            print("📊 Generating Topological Diagrams (Sankey/Chord) for 3D data...")
            av = AdvancedVisualizer(adata, cfg)
            
            # Figure 6: Chord Diagram
            av.plot_chord_diagram(
                matrix_for_3d, 
                save_path=os.path.join(cfg.result_dir, "Figure_6_Chord_Circlize.pdf")
            )
            # Figure 8: Sankey Diagram
            av.plot_sankey_flow(
                matrix_for_3d, 
                save_path=os.path.join(cfg.result_dir, "Figure_8_Sankey.pdf")
            )
            # Figure 5: Interaction Matrix Heatmap
            av.plot_cluster_interaction_matrix(
                results_df, 
                value_col='strength',
                save_path=os.path.join(cfg.result_dir, "Figure_5_Interaction_Matrix.pdf")
            )
        
        # 3. 3D Hotspots
        t3d.plot_3d_interaction_hotspot(final_results, save_path=os.path.join(cfg.result_dir, "Figure_3_3D_Hotspot.pdf"))
        
        # 4. DotPlot
        print("🫧 Generating DotPlot...")
        av = AdvancedVisualizer(adata, cfg)
        av.plot_lr_dotplot(final_results, save_path=os.path.join(cfg.result_dir, "Figure_7_LR_DotPlot.pdf"))

    else:
        # =================================
        # 2D Pipeline (Standard)
        # =================================
        print("🗺️ Entering 2D Visualization Pipeline...")
        
        if len(final_results) > 0:
            Visualizer.plot_interaction_spatial(
                adata, final_results[0], 
                save_path=os.path.join(cfg.result_dir, "Figure_3A_Example.pdf")
            )
            Visualizer.plot_lr_colocalization_heatmap(
                adata, final_results[0],
                save_path=os.path.join(cfg.result_dir, "Figure_3B_Halo.pdf")
            )

        if results_df is not None:
            av = AdvancedVisualizer(adata, cfg)
            matrix = av.plot_cluster_interaction_matrix(
                results_df, 
                value_col='strength',
                save_path=os.path.join(cfg.result_dir, "Figure_5_Interaction_Matrix.pdf")
            )

            if matrix is not None:
                av.plot_spatial_network_dark(
                    matrix, 
                    save_path=os.path.join(cfg.result_dir, "Figure_1_Spatial_Network_Dark.pdf"),
                    background_type='celltype'
                )
                av.plot_chord_diagram(
                    matrix, 
                    save_path=os.path.join(cfg.result_dir, "Figure_6_Chord_Circlize.pdf")
                )
                av.plot_sankey_flow(
                    matrix, 
                    save_path=os.path.join(cfg.result_dir, "Figure_8_Sankey.pdf")
                )
        
        av = AdvancedVisualizer(adata, cfg)
        av.plot_lr_dotplot(final_results, save_path=os.path.join(cfg.result_dir, "Figure_7_LR_DotPlot.pdf"))

    print("\n✅ All experiments completed! Please check the output folder.")