---
published: false
---
## Pretraining multimodal (vision and language) transformers with pytorch lightning

Self supervised pretraining of transformers is a default component to training strong performing vision and language models. Research typically follows the following setup:

![Overview of SSL]({{site.baseurl}}/images/pretraining-transformers/ssl_overview.png)

There have been many approaches using a variety of pretext tasks (i.e. the method to supervise the model using an unlabelled dataset) and I was curious how, holding everything else constant, these pretext tasks compared on a baseline model. Several excellent repos exist such as LXMERT, VisualBert, and UNITER, however to make running and tracking experiments easier I implemented my own framework within pytorch lightning (relying on the HF transformers library for the model backbone). This post documents this process.

This post comprises several sections:

1. Visual feature extraction & dataset building
2. Framework architecture
3. Pretraining tasks
4. Running (pretraining) experiments with weights & biases

### Overview

![composite pretext tasks]({{site.baseurl}}/images/pretraining-transformers/pretext_tasks.png)

