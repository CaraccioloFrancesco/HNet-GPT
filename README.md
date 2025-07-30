# HNet-GPT
# HNet-GPT: Structure-Aware Code Generation via Hierarchical Encoding and Transformer Decoding

**Exploratory Research** - Hierarchical encoding for structure + GPT-2 for generation. 
Seeking community feedback and collaboration




## The idea
Large Language Models (LLMs) like GPT-2 have demonstrated strong performance in code generation, but they treat source code as a flat sequence of tokens, ignoring its rich structural and hierarchical nature. In this work, I  propose a lightweight hybrid architecture that enhances code generation by combining a hierarchical chunk-based encoder (H Net) with a pretrained autoregressive decoder (GPT-2). This model leverages structural priors by encoding code into semantically coherent segments using similarity-driven dynamic chunking, routed through a Transformer-based encoder, and decoded using GPT 2’s language modeling head.

•	Problem: Existing code generation models (e.g., GPT) tokenize flatly, ignoring structural hierarchy.
•	Motivation: Human code understanding involves structural chunking and semantic abstraction.
•	Proposal: Combine H-Net encoder (chunk-aware) with a GPT decoder (pretrained generation) — called HNet GPT.
•	Contributions:
o	New hybrid architecture for code modeling
o	Empirical improvements in perplexity over baseline GPT-2
o	Open-source training pipeline and reproducibility



## The questions
- Can hierarchical encoding improve code understanding?
- How do we best combine structural and sequential information?
- What are the limits of small-scale transformer training?

## Current Results (Preliminary) and the story
- Small-scale experiments show promise
- Need validation on larger datasets
- Seeking collaboration for scaling up

- Your Research Story:

Chapter 1: "We found hierarchical encoding helps (+40% from scratch)"
Chapter 2: "It also works as efficient adapters (+55% with 7.9% params)"
Chapter 3: "Future work: Optimize for generation quality"


## Limitations & Future Work
Currently the training is limited and done on a small dataset. The programming language used is python with its adaptive chunking based on code syntax (e.g., def, class, if) is smart and aligns with how code is semantically structured.

## Call for Collaboration
I am  open-sourcing everything to:
- Start collaborations
- Enable reproducibility
- Gather community feedback
- Validate results across different setups
- Explore further improvements on the architecure and the experiment design

## Preliminary Findings
but to be rerunned
- **40.6%** improvement over Pure GPT-2
- **39.5%** improvement over Pure HNet
- Comprehensive evaluation on MBPP dataset

## Next Steps
- [ ] Scale to larger models --> Connect to a more recent GPT
- [ ] Test on more datasets --> train on C/C++
- [ ] Optimize architecture
- [ ] Community feedback integration


## How to use it 

full end-to-end benchmark framework refer to the notebook (baselines, training, evaluation, and analysis)
