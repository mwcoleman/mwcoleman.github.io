---
published: true
---
## Pretraining multimodal (vision and language) transformers with pytorch lightning

As a component of my current research project using transformers for multimodal radiology data, I've been working to implement a framework for evaluating pretext tasks used to pretrain multimodal transformers. . This post provides a walkthrough of the steps I've taken so far.

1. Overview
2. Visual feature extraction & dataset building
3. Framework architecture
4. Pretraining tasks
5. Running (pretraining) experiments with weights & biases


Note: The following resouces were used in various ways:
- LXMERT
- UNITER
- VisualBert
- Huggingface Transformers

### Overview

Self supervised pretraining of transformers is a default component to training strong performing vision and language models. Research typically follows the following setup:



There have been many approaches using a variety of pretext tasks (i.e. the method to supervise the model using an unlabelled dataset) and I was curious how, holding everything else constant, these pretext tasks compared on a baseline model.

VisualBert was chosen as the baseline model as it is a simple achitecture with decent performance, and there is a huggingface implementation.
