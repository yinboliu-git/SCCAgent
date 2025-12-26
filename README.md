# SCCAgent
A Spatial Cell-Cell Communication Intelligent Agent for Decoding Intercellular Signaling in Spatial Transcriptomics Data

# SpaceAgent: Spatially-Aware LLM Framework for Gene Set Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/LLM-GPT--4-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SpaceAgent** represents a novel framework that integrates Large Language Models (LLMs) with spatial transcriptomics data. Unlike traditional methods, SpaceAgent utilizes **GPT-4** to reason about ligand-receptor interactions by combining biological knowledge (OmniPath, KEGG, GO) with physical spatial topology.

This repository contains the implementation for the API-based version, optimized for **OpenAI GPT-4**.

---

## 🌟 Key Features

* **Spatial-Logic Reasoning**: Filters false-positive interactions by enforcing spatial distance constraints (Distance Decay + Mass Action laws).
* **Knowledge Graph Injection**: Automatically retrieves and injects background knowledge from OmniPath, KEGG, and Reactome into the LLM prompt.
* **Hallucination Detection**: Built-in modules to statistically verify LLM outputs against random baselines.
* **Advanced Visualization**:
    * **2D/3D Network Graphs**: Spatially-embedded interaction networks.
    * **Topological Plots**: Sankey diagrams and Chord diagrams for intercellular communication flows.
    * **3D Rendering**: Native support for 3D datasets (e.g., human embryo) with "nebula-style" visualization.
* **GPT-4 Powered**: Uses the state-of-the-art reasoning capabilities of GPT-4 for biological validation.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.9+** installed.
2.  **OpenAI API Key**: You need a valid API key with access to `gpt-4` or `gpt-4-turbo`.
    * [Get your API Key here](https://platform.openai.com/api-keys)

---

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YourUsername/SpaceAgent.git](https://github.com/YourUsername/SpaceAgent.git)
    cd SpaceAgent
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment (Conda or venv).
    ```bash
    pip install -r requirements.txt
    ```

    *If `requirements.txt` is missing, install the core packages manually:*
    ```bash
    pip install numpy pandas scanpy squidpy matplotlib seaborn networkx scipy openai pycirclize plotly
    ```

---

## ⚙️ Configuration (Crucial)

To use SpaceAgent with GPT-4, you must configure the `SpaceConfig` in your execution script (e.g., `run_MouseBrain.py`).

Open `run_MouseBrain.py` and modify the configuration block inside the `__main__` section:

```python
if __name__ == "__main__":
    # ...
    cfg = SpaceConfig(
        data_dir="./gold_data",
        result_dir="./results_gold/custom_experiment/",
        
        # === LLM Settings ===
        llm_source="api",   # MUST be set to 'api'
        
        # === OpenAI Credentials ===
        openai_api_key="sk-YourOpenAIKeyHere...", # <--- REPLACE THIS
        openai_model="gpt-4",  # Recommended: "gpt-4" or "gpt-4-turbo"
        
        # Other settings...
        gpu_id=0 # GPU is used for data processing (Scanpy), not for LLM inference
    )
