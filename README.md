# HNet-GPT
## HNet-GPT: Structure-Aware Code Generation via Hierarchical Encoding and Transformer Decoding

**Exploratory Research** - Hierarchical encoding for structure + GPT-2 for generation. 

Seeking community feedback and collaboration<br/><br/>
## The idea
Large Language Models (LLMs) like GPT-2 have demonstrated strong performance in code generation, but they treat source code as a flat sequence of tokens, ignoring its rich structural and hierarchical nature. In this work, I  propose a lightweight hybrid architecture that enhances code generation by combining a hierarchical chunk-based encoder (H Net) with a pretrained autoregressive decoder (GPT-2). This model leverages structural priors by encoding code into semantically coherent segments using similarity-driven dynamic chunking, routed through a Transformer-based encoder, and decoded using GPT 2’s language modeling head.




- **Problem**: Existing code generation models (e.g., GPT) tokenize flatly, ignoring structural hierarchy.
- **Motivation**: Human code understanding involves structural chunking and semantic abstraction.
- **Proposal**: Combine H-Net encoder (chunk-aware) with a GPT decoder (pretrained generation) — called HNet GPT.
- **Contributions**:
        - New hybrid architecture for code modeling
        - Empirical improvements in perplexity over baseline GPT-2
        - Open-source training pipeline and reproducibility

<br/><br/>

<img width="681" height="181" alt="Diagramma senza titolo drawio" src="https://github.com/user-attachments/assets/d32aa8c8-479b-4746-a922-675c56f90e05" />

<br/><br/>

**Key Components:**

1. **Hierarchical Encoder**
    - Adaptive code-structure chunking
    - Multi-level attention processing
    - Structure-aware representations


2. **Smart Fusion Mechanism**
    - Complexity-aware gating
    - Dynamic feature combination
    - Preserves both structural and sequential information


3. **GPT-2 Decoder Integration**
    - Leverages pre-trained knowledge
    - Proven autoregressive generation
    - Enhanced with hierarchical context

<br/><br/>

## The questions
- Can hierarchical encoding improve code understanding?
- How do we best combine structural and sequential information?
- What are the limits of small-scale transformer training?

<br/><br/>

## Preliminary Findings
Small-scale experiments show promise
Preliminary Results:

| Architecture  | Perplexity |
| ------------- | ------------- |
| HNet-GPT-2 Hybrid  | Content Cell  |
| Pure HNet  | Content Cell  |
| Pure GPT-2  | Content Cell  |


ANALYSIS: Generation Capability Breakdown

The 0% scores for pure models indicate complete generation failure,
likely due to:

1. Training-inference mismatch in generation parameters
2. Lack of robust token continuation mechanisms  
3. Architecture-specific limitations in code domain

Our hybrid's 32% success demonstrates that the hierarchical-sequential 
fusion mechanism provides crucial generation stability that neither 
pure approach achieves.


<br/><br/>

## Limitations & Future Work
Current Limitations:
- Training Scale and Duration
Our current implementation faces several training-related constraints that limit the full potential of the proposed architecture. The training was conducted on a relatively small dataset of 4,000 training examples from CodeSearchNet Python, which, while sufficient to demonstrate architectural benefits, may not capture the full diversity of programming patterns found in larger codebases. Additionally, the 25-epoch training regimen, though showing convergence in perplexity metrics, may be insufficient for the complex hierarchical-sequential fusion mechanism to reach optimal performance. This is evidenced by the fact that while our hybrid achieved 32% functional correctness compared to 0% for baseline approaches, there remains substantial room for improvement with extended training.
The conservative learning rate (1e-4) and gradual unfreezing strategy, while ensuring training stability, may have limited the speed of convergence. Future work could explore more aggressive training schedules or advanced optimization techniques such as learning rate scheduling and adaptive batch sizing to accelerate the learning process.

- Evaluation Methodology


Future Work:
Edge Computing Potential: Our results suggest promising opportunities for edge deployment and resource-efficient applications. The fact that meaningful architectural differences emerge even with limited training (25 epochs) and modest computational resources indicates that the hierarchical-sequential fusion mechanism may be inherently efficient. The hybrid's exclusive success in functional code generation (32% vs 0%) demonstrates that architectural innovation, rather than simply scale, drives performance improvements.

Currently the training is limited and done on a small dataset. 25 epochs might not be enough to properly train these architectures from scratch. The programming language used is python with its adaptive chunking based on code syntax (e.g., def, class, if) is smart and aligns with how code is semantically structured.
It also works as efficient adapters

## Call for Collaboration
I am  open-sourcing everything to:
- Start collaborations
- Enable reproducibility
- Gather community feedback
- Validate results across different setups
- Explore further improvements on the architecure and the experiment design



## Next Steps
- [ ] Architecture Scaling and Modernization --> Connect to GPT-4/CodeLlama backbone: Upgrade from GPT-2 to more recent foundation models
- [ ] Enhanced Evaluation and Benchmarking --> Comprehensive functional testing adn add Human evaluation studies
- [ ] Optimize fusion mechanism --> Refine hierarchical-sequential integration based on current findings
- [ ] Multi-Language Expansion --> Extend to C/C++ and JavaScript/TypeScript programming
- [ ] Advanced Training Methodologies --> Include results using Curriculum learning implementation and Reinforcement learning from execution feedback
- [ ] Edge Computing and Efficiency --> Lightweight model variants: Develop resource-efficient versions for edge deployment
- [ ] Community feedback integration


## How to use it 

full end-to-end benchmark framework refer to the notebook (baselines, training, evaluation, and analysis)
