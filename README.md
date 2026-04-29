# DE-RAG

**Differential Evolution Enhanced Offline Optimization for Knowledge Graph based Retrieval Augmented Generation**

## Overview

DE-RAG is an offline optimization framework designed to improve knowledge graph based retrieval augmented generation systems. It optimizes both knowledge graph structure weights and graph search behavior to enhance downstream question answering performance.

Unlike traditional approaches that rely on fixed graph configurations, DE-RAG formulates retrieval adjustment as a continuous optimization problem and solves it using differential evolution.

## Key Idea

DE-RAG jointly optimizes:

- Relation cluster scaling factors for knowledge graph edges  
- Graph search hyperparameters that control retrieval behavior  

Optimization is performed through a retrieval and question answering loop, using F1 score as the objective.

## Method

The framework consists of three main components:

### 1. Knowledge Graph Construction
- Extract entities and triples from corpus  
- Build a multi edge knowledge graph  
- Initialize edge weights based on occurrence frequency  

### 2. Parameterization
- Cluster relation edges into semantic groups  
- Assign a learnable scaling factor to each cluster  
- Introduce graph search parameters:
  - reset weight  
  - damping factor  

### 3. Optimization
- Encode parameters into a unified vector  
- Apply differential evolution  
- Evaluate candidates via retrieval and QA loop  

## Features

- Joint optimization of graph structure and retrieval behavior  
- Adaptable to different corpus sizes and query types  
- Compatible with existing knowledge graph based RAG systems  
- Supports warm start for efficient re-optimization  

## Experiments

Evaluated on:

- 2WikiMultihopQA  
- HotpotQA  
- MuSiQue  
- NarrativeQA  

DE-RAG improves F1 score over baseline configurations while maintaining competitive retrieval performance.

## Usage

```bash
# build knowledge graph
python build_kg.py

# run optimization
python optimize.py

# run evaluation
python evaluate.py
