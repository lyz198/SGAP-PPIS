# SGAP-PPIS：Structure-Guided Adaptive Propagation for Protein-Protein Interaction Site Prediction

## Abstract
Motivation: Accurate prediction of protein-protein interaction sites (PPIS) is essential for understanding cellular processes, disease mechanisms, and therapeutic target discovery. Graph-based deep learning has advanced PPIS prediction by incorporating residue-level structural context. However, most graph-based models still rely on fixed propagation schemes that treat all residues similarly, despite the structural and functional heterogeneity of protein interfaces. Such propagation may limit the ability to adapt information diffusion to local geometric environments, making it difficult to distinguish true interaction sites from structurally similar non-interacting neighbors.
Results: We present SGAP-PPIS, a structure-guided adaptive propagation model for PPIS prediction. Rather than using a fixed propagation mechanism, SGAP-PPIS leverages multi-scale geometric states from an equivariant graph
neural network to generate residue-wise propagation coefficients. This design allows each residue to adaptively balance local feature preservation and neighborhood diffusion according to its geometric microenvironment. Experimental results show that SGAP-PPIS achieves competitive performance among the state-of-the-art methods on Test_60. Ablation studies show that geometry-conditioned adaptive propagation, scale-aligned geometric
guidance, and multi-step propagation-state representation jointly drive these improvements. Availability and implementation: The datasets and source code are available at https://github.com/lyz198/SGAP-PPIS.

## Preparation
### Environment Setup
The repo mainly requires the following packages.
+ dgl==2.4.0+cu118
+ numpy==1.26.4
+ pandas==2.2.3
+ scikit-learn==1.5.2
+ scipy==1.15.3
+ sympy==1.14.0
+ torch==2.3.1+cu118
+ torchdata==0.7.1




