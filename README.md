# HNet-GPT
# HNet-GPT: Structure-Aware Code Generation via Hierarchical Encoding and Transformer Decoding

**Exploratory Research** - Hierarchical encoding for structure + GPT-2 for generation. 
Seeking community feedback and collaboration

## The idea
Presentation of a novel architecture HNet-GPT with the goal of optimization Code understanting and generation tasks. This architecture uses hierchical encoding for structure and GPT-2 for generation --> improve this part 

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
- Enable reproducibility
- Gather community feedback
- Validate results across different setups
- Explore further improvements

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
