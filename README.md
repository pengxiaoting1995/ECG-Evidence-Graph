# ECG-Evidence-Graph

ECG-Evidence-Graph is a graph-based framework for prolonged ECG analysis under extreme sparsity. The framework performs reliability-aware beat selection, multi-criteria evidence-node construction, topology-aware graph reconstruction, and patient-level graph classification for disease prediction from long-term ECG recordings.

## Framework overview

- `ccr_score/` : beat-level reliability learning
- `pruning_recovery/` : multi-criteria node-set construction
- `graph_construction/` : topology-aware graph reconstruction
- `graph_model/` : graph classification and meta-classifier refinement

## Pipeline

1. Beat-level reliability learning
2. Multi-criteria node-set construction
3. Topology-aware graph reconstruction
4. Graph classification
5. Meta-classifier refinement

## Data format

- Beat-level HDF5
- Patient-level graph CSV

## Example workflow

```bash
ccr_score/train.py
pruning_recovery/pipeline.py
graph_construction/build_graph.py
graph_model/train.py
graph_model/meta_classifier.py
graph_model/infer.py