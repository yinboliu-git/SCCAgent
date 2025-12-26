import os
import importlib
import logging
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # For calculating inter-cluster distances

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
    AdvancedVisualizer
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Core Function Encapsulation
# ==========================================

def build_global_network_data(adata, final_results, cfg):
    """
    [Core Fix] Build global interaction network data (CellChat style - Law of Mass Action + Distance Decay)
    
    Improvements:
    1. Introduce Law of Mass Action: Score = Avg(L)_src * Avg(R)_tgt
    2. Introduce Distance Decay: Closer distances result in higher weights, instead of a simple 0/1 cutoff.
    """
    logger.info("🔄 Calculating Weighted Source-Target interactions (Mass Action + Distance Decay)...")
    
    # 1. Prepare data
    if 'target_cluster' not in adata.obs:
        return None

    adata.obs['target_cluster'] = adata.obs['target_cluster'].astype(str)
    unique_clusters = adata.obs['target_cluster'].unique()
    
    # 2. Pre-compute Mean Expression for all Clusters
    # This step is crucial for calculating L*R strength later
    logger.info("   - Pre-computing cluster gene expression profiles (Mean Expression)...")
    cluster_expr_means = {}
    
    # To accelerate, convert to DataFrame or operate directly on numpy
    # Note: Assumes data is already normalized (log1p); if it's counts, normalization is needed first
    for cid in unique_clusters:
        mask = adata.obs['target_cluster'] == cid
        if mask.sum() == 0: continue
        
        # Get mean expression for this Cluster
        # Handle sparse matrices
        X_subset = adata.X[mask]
        if hasattr(X_subset, "toarray"):
            mean_expr = np.array(X_subset.mean(axis=0)).flatten()
        else:
            mean_expr = np.array(X_subset.mean(axis=0)).flatten()
            
        # Store in dictionary, Key is gene name, Value is expression value
        cluster_expr_means[cid] = pd.Series(mean_expr, index=adata.var_names)

    # 3. Prepare spatial calculation parameters
    if 'spatial_norm' in adata.obsm:
        coords_all = adata.obsm['spatial_norm']
        dist_thresh = getattr(cfg, 'adaptive_threshold', 0.05)
        logger.info(f"   - Using Normalized Coords (Threshold={dist_thresh:.4f})")
    else:
        coords_all = adata.obsm['spatial']
        dist_thresh = 250.0

    # 4. Build edge list
    matrix_data = []
    
    for entry in final_results:
        src_cluster = str(entry['cluster_id'])
        txt = entry.get('llm_analysis', '')
        
        # Iterate through interactions deemed valid by LLM
        for pair in entry.get('known_interactions', []):
            g1, g2 = pair.split(' -> ') # g1=Ligand, g2=Receptor
            
            # A. Must pass LLM semantic verification
            if (g1 in txt and g2 in txt) and ("Segregated" not in txt):
                
                # Get Ligand expression for Source
                if src_cluster not in cluster_expr_means: continue
                val_L = cluster_expr_means[src_cluster].get(g1, 0.0)
                
                # Skip if Source does not express this ligand (or expression is very low)
                if val_L < 0.01: continue 

                # Get Source coordinates
                mask_src = adata.obs['target_cluster'] == src_cluster
                coords_src = coords_all[mask_src]
                
                # Iterate through all Target Clusters
                for tgt_cluster in unique_clusters:
                    # B. Get Receptor expression for Target
                    if tgt_cluster not in cluster_expr_means: continue
                    val_R = cluster_expr_means[tgt_cluster].get(g2, 0.0)
                    
                    # If Target does not express receptor, communication strength is 0
                    if val_R < 0.01: continue
                    
                    # === Core Improvement 1: Law of Mass Action ===
                    # Base Strength = Ligand_Expr * Receptor_Expr
                    # This ensures Clusters with high receptor expression get higher weights
                    base_strength = val_L * val_R
                    
                    min_dist = float('inf')
                    
                    # C. Spatial Distance Calculation
                    if src_cluster == tgt_cluster:
                        # Autocrine: Distance treated as 0, highest weight
                        min_dist = 0.0
                        dist_factor = 1.0
                    else:
                        mask_tgt = adata.obs['target_cluster'] == tgt_cluster
                        coords_tgt = coords_all[mask_tgt]
                        
                        if len(coords_tgt) > 0:
                            # Calculate minimum distance
                            dists = cdist(coords_src, coords_tgt)
                            min_dist = dists.min()
                            
                            # === Core Improvement 2: Distance Decay (Soft Threshold) ===
                            if min_dist > dist_thresh:
                                dist_factor = 0.0 # Exceeds threshold, cutoff
                            else:
                                # Linear decay: closer distance (0) -> factor approaches 1; distance near threshold -> factor approaches 0
                                dist_factor = 1.0 - (min_dist / dist_thresh)
                                # Or use exponential decay (smoother): 
                                # dist_factor = np.exp(-min_dist / (dist_thresh * 0.5))
                        else:
                            dist_factor = 0.0

                    # D. Final Score
                    final_score = base_strength * dist_factor
                    
                    if final_score > 0:
                        matrix_data.append({
                            'source': src_cluster, 
                            'target': tgt_cluster, 
                            'count': 1,              # Keep count for reference
                            'strength': final_score, # [New] Real physicochemical strength
                            'lr_pair': pair
                        })
    
    if not matrix_data:
        return None

    return pd.DataFrame(matrix_data)

# ==========================================
# Main Program Entry Point
# ==========================================

# ==========================================
# Main Program Entry Point (Modified for GPT-4 Support)
# ==========================================

if __name__ == "__main__":
    # 1. Initialize Configuration
    data_name = 'squidpy_seqfish'
    print("🚀 Initializing SpaceAgent Framework...")
    
    # === [Key Modification] Configure LLM Mode and Keys Here ===
    cfg = SpaceConfig(
        data_dir="./gold_data", 
        result_dir=f"./results_gold/{data_name}/",
        llm_source="api", 
        openai_api_key="sk-XXXXXXXXXXXXXXXXXXXXXXXX",  # <--- Please enter your real key here
        openai_model="gpt-4", 
        gpu_id=0
    )

    # Initialize modules
    dm = DataManager(cfg)
    sa = SpaceAnalyzer(cfg)
    agent = AgentInterface(cfg) # Automatically reads cfg.llm_source to decide which model to load

    # 2. Data Preparation
    print("\n📦 Preparing Data...")
    dm.prepare_knowledge_base()
    h5ad_path = dm.load_dataset(source=data_name) 

    # 3. Preprocessing & Topology Analysis
    print("\n🔬 Preprocessing & Analyzing Topology...")
    json_data, adata = dm.preprocess_adata(h5ad_path)
    
    if 'target_cluster' in adata.obs:
        print(f"✅ Using Cluster Column: {adata.obs['target_cluster'].name}")
        print(f"   Clusters: {adata.obs['target_cluster'].unique().tolist()}")
    else:
        print("⚠️ Warning: No cluster column found.")

    # Generate topology report (inject functional annotations)
    json_data = sa.generate_reports(json_data, adata, data_manager=dm)
    
    # 4. LLM Inference (Agent will automatically call GPT-4 based on cfg)
    print(f"\n🤖 Running LLM Inference (Mode: {cfg.llm_source})...")
    agent.load_model() # Initializes OpenAI Client if in API mode
    final_results = agent.run_inference(json_data)
    
    # Backup inference results
    pd.DataFrame(final_results).to_json(
        os.path.join(cfg.result_dir, "raw_inference_results.json")
    )

    # ... (Subsequent code remains unchanged: Hallucination test, Visualization, etc.) ...
    # 5. Hallucination Test (Figure 4)
    print("\n📉 Running Hallucination Evaluation (Figure 4)...")
    hallucination_checker = HallucinationAnalyzer(sa)
    dist_data = hallucination_checker.collect_distances(adata, final_results)

    Visualizer.plot_hallucination_test(
        dist_data, 
        save_path=os.path.join(cfg.result_dir, "Figure_4_Hallucination.pdf")
    )

    # 6. Quantitative Benchmark (Figure 5 & Table)
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

    # 7. Qualitative Evaluation (Table 3) - Automatically uses GPT-4 as Judge
    print("\n👩‍⚖️ Running Qualitative Quality Evaluation (Table 3)...")
    judge = QualityEvaluator(cfg)
    quality_df = judge.evaluate_fairness(final_results)
    if not quality_df.empty:
        quality_df.to_csv(os.path.join(cfg.result_dir, "quality_scores.csv"), index=False)

    # 8. Advanced Visualization
    print("\n🎨 Generating Publication-Ready Figures...")

    # 8.1 Basic Plots
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

    # 8.2 Global Network Graph Construction
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
            
            # If a version with H&E background is needed (Valid only for Visium data)
            # av.plot_spatial_network_light(
            #     matrix, 
            #     save_path=os.path.join(cfg.result_dir, "Figure_1_Spatial_Network_Light_imgs.pdf"),
            #     background_type='image'
            # )
            
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