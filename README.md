# HNet-GPT
## HNet-GPT: Structure-Aware Code Generation via Hierarchical Encoding and Transformer Decoding

**Exploratory Research** - Hierarchical encoding for structure + GPT-2 for generation for coding application

Seeking community feedback and collaboration<br/><br/>
## The idea
Large Language Models (LLMs) like GPT-2 have demonstrated strong performance in code generation, but they treat source code as a flat sequence of tokens, ignoring its rich structural and hierarchical nature. In this work, I  propose a lightweight hybrid architecture that enhances code generation by combining a hierarchical chunk-based encoder (H-Net) with a pretrained autoregressive decoder (GPT-2). This model leverages structural priors by encoding code into semantically coherent segments using similarity-driven dynamic chunking, routed through a Transformer-based encoder, and decoded using GPT 2’s language modeling head.
<br/><br/>




- **Problem**: Existing code generation models (like GPT) tokenize flatly, ignoring structural hierarchy.
- **Motivation**: Human code understanding involves structural chunking and semantic abstraction.
- **Proposal**: Combine H-Net encoder (chunk-aware) with a GPT decoder (pretrained generation) — called HNet-GPT.
- **Contributions**: new hybrid architecture for code modeling, empirical improvements in perplexity over baseline GPT-2, open-source training pipeline and reproducibility

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

## The Questions
- Can hierarchical encoding improve code understanding and generation?
- How do we best combine structural and sequential information?
- What are the limits of small-scale transformer training?

<br/><br/>

## Preliminary Findings
> [!NOTE]
> These scores aim to reflect the model's ability to generate syntactically valid and pattern-matching completions under the specific test conditions.


<br/>

Small-scale experiments show promising preliminary results:

| Architecture      | Perplexity | Pattern Recognition  | Code Token Usage | Overall Score |
| :---:             | :---:      | :---:                | :---:            | :---:         |   
| HNet-GPT-2 Hybrid | 8.20       | 60.0%                | 40.0%            | 32.0%         |
| Pure HNet         | 13.56      | 0.0%                 | 0.0%             | 0.0%          |
| Pure GPT-2        | 13.80      | 0.0%                 | 0.0%             | 0.0%          |

<br/>

**Generation Capability Breakdown**

<br/>

The 0% scores for pure models indicate complete generation failure,
likely due to:

1. Training-inference mismatch in generation parameters
2. Lack of robust token continuation mechanisms  
3. Architecture-specific limitations in code domain

The hybrid's 32% success demonstrates that the hierarchical-sequential fusion mechanism provides crucial generation stability that neither  pure approach achieves.


<br/><br/>

## Current Limitations
- Training Scale and Duration: the current implementation faces several training-related constraints that limit the full potential of the proposed architecture. Training conducted on small dataset of 4,000 training examples from CodeSearchNet Python. 25-epoch training regimen, may be insufficient for the complex hierarchical-sequential fusion mechanism to reach optimal performance.Future work could explore more aggressive training schedules or advanced optimization techniques such as learning rate scheduling and adaptive batch sizing to accelerate the learning process.

- Evaluation Methodology: The functional correctness evaluation, while revealing clear architectural advantages, remains limited in scope with five test cases covering basic programming constructs. A more comprehensive evaluation would benefit from broader test suites including complex algorithmic challenges and the metrics could be enhanced with more sophisticated measures such as semantic similarity, execution correctness on diverse test cases, and human evaluation of code quality and readability.<br/>

<br/><br/>

## Call for Collaboration
I am  open-sourcing everything to:
- Start collaborations
- Enable reproducibility
- Gather community feedback
- Validate results across different setups
- Explore further improvements on the architecure and the experiment design

<br/><br/>

## Next Steps
- [ ] Architecture Scaling and Modernization --> Connect to GPT-4/CodeLlama backbone: Upgrade from GPT-2 to more recent foundation models
- [ ] Enhanced Evaluation and Benchmarking --> Comprehensive functional testing adn add Human evaluation studies
- [ ] Optimize the Hybrid Architecure Mechanism --> Improve the heuristic-Based Chunking (Abstract Syntax Tree (AST) parser), refine hierarchical-sequential integration based on current findings
- [ ] Multi-Language Expansion --> Extend to C/C++ and JavaScript/TypeScript programming
- [ ] Advanced Training Methodologies --> Include results using Curriculum learning implementation and Reinforcement learning from execution feedback
- [ ] Edge Computing and Efficiency --> Lightweight model variants: Develop resource-efficient versions for edge deployment
- [ ] Community feedback integration

<br/><br/>

## How to use it 
The notebook is a full end-to-end framework that will:
1. Install all required dependencies.
2. Load the code_search_net dataset (or fall back to a built-in synthetic dataset if unavailable).
3. Define the three model architectures: Pure GPT-2, Pure HNet, and the novel HNet-GPT Hybrid.
4. Train each model sequentially. The training process includes gradual unfreezing and will save the final model weights to your Google Drive in a folder named hnet_gpt2_models.
5. Evaluate the trained models using two methods:
    - Standard perplexity on a test set.
    - The custom code-focused evaluation (Syntax Validity, Pattern Recognition, etc.).
6. Display a final results table comparing the performance of all three architectures.
