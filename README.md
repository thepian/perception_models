# Perception Models: Powerful Models for Image and Video Perception

**[Meta AI Research, FAIR](https://ai.facebook.com/research)**

Perception Models is home to the state-of-the-art for image and video _perception_: Perception Encoder (PE) for encoding and Perception Language Model (PLM) for decoding. We designed Perception Models as a user-friendly repository to support the training, inference, and evaluation of these two models, with an emphasis on making the code modular and easy to expand and experiment with.


* **[Apr-18-25]:** Perception Language Model (PLM) and PLM-VideoBench are added to lmms-eval. This makes it easy to reproduce PLM results and allows you to evaluate on the PLM-VideoBench. [[`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/638)] :fire::fire:
* **[Apr-17-25]:** Perception Encoder (PE) and Perception Language Model (PLM) are released. [[`Blog`](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning)] :fire::fire:


## Perception Encoder (PE)

[![Paper](https://img.shields.io/badge/Paper-Perception%20Encoder-b31b1b.svg)](https://ai.meta.com/research/publications/perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the-network) 
[![Paper](https://img.shields.io/badge/arXiv-2504.13181-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2504.13181)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/facebook/perception-encoder-67f977c9a65ca5895a7f6ba1)
[![Colab](https://img.shields.io/badge/Google%20Colab-Tutorials-red)](apps/pe/docs/pe_demo.ipynb)
[![Model License](https://img.shields.io/badge/Model_License-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

Perception Encoder (PE) is an extremely powerful and versatile family of vision encoders for both images and video: PE core can outperform SigLIP2 on Image CLIP and InternVideo2 on Video CLIP; PE lang can be used to outperform QwenVL2.5 and InternVL3 on vision language modeling; and PE spatial can outperform DINOv2 on dense prediction tasks. And all of this follows the same, easily scalable contrastive pretraining.

See [`apps/pe/README.md`](apps/pe/README.md) for more information and how to get started using them!




## Perception Language Model (PLM)
[![Paper](https://img.shields.io/badge/Paper-PerceptionLM-b31b1b.svg)](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding)
[![Paper](https://img.shields.io/badge/arXiv-2504.13180-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2504.13180)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-blue)](https://huggingface.co/collections/facebook/perception-lm-67f9783f171948c383ee7498)
[![Colab](https://img.shields.io/badge/Google%20Colab-Tutorials-red)](apps/plm/notebook_demos)
[![ModelLicense](https://img.shields.io/badge/Model_License-FAIR_Research_License-lightgrey)](../../LICENSE.PLM)

PerceptionLM (PLM) is a family of open and fully reproducible models to facilitate research in vision-language modeling (VLM). In conjunction with PE, it is powerful enough to compete with the latest state-of-the-art VLMs such as InternVL3 and QwenVL2.5, while using _fully open data_. We also release the largest spatiotemporally annotated video dense captioning and fine-grained human activity recognition datasets to ever exist.

**Overview**: PLM consists of a vision encoder with a small scale (<8B parameters) LLM decoder. We start by an analysis of standard training pipelines with available data, without any proprietary model distillation. We investigate large-scale synthetic data and establish key scaling laws to identify critical data gaps that limit video understanding performance, especially for spatio-temporal reasoning and fine-grained understanding tasks. To fill these gaps, we release 2.8M human-labeled instances of fine-grained video question-answer pairs and spatio-temporally grounded video captions. This release is nearly an order of magnitude larger than the largest existing video datasets. Additionally, we introduce PLM–VideoBench, a suite for evaluating challenging video understanding tasks focusing on the ability to reason about "what", "where", "when", and "how" of a video.

![Description of the image](apps/plm/docs/plm_main_fig.png)


## PLM Image Benchmark Results

| Model  | DocVQA | ChartQA | TextVQA | InfoQA | AI2D  | OCRBench | COCO | Nocap | Flickr | MMMU | VQAv2 | OKVQA | VizWiz | MME | SEED | BLINK | CVBench | RealWorldQA | VSR | POPE |
|:---------:|:--------:|:---------:|:---------:|:--------:|:------:|:----------:|:------------:|:-------------:|:--------------:|:------:|:-------:|:--------:|:--------:|:-----:|:------:|:-------:|:----------:|:-------------:|:-----:|:------:|
| PLM1B  | 90.7   | 78.6    | 82.1    | 63.0   | 84.9 | 807      | 138.6      | 124.2       | 100.5        | 34.8 | 81.7  | 61.0   | 59.7   | 1603| 76.3 | 46.8  | 73.8     | 67.1        | 68.8| 88.4 |
| PLM3B  | 93.8   | 84.3    | 84.3    | 74.6   | 90.9 | 830      | 144.9      | 126.5       | 98.0         | 41.2 | 84.3  | 66.8   | 64.0   | 1879| 78.5 | 55.4  | 81.4     | 72.4        | 80.4| 88.7 |
| PLM8B  | 94.6   | 85.5    | 86.5    | 80.9   | 92.7 | 870      | 146.7      | 129.9       | 105.6        | 46.1 | 85.6  | 69.6   | 67.0   | 1989| 79.3 | 56.0  | 81.3     | 75.0        | 82.8| 89.9 |

## PLM Video Benchmark Results

| Model  | VATEX                    | DREAM&nbsp;1K      | How2QA       | MVBench      | NExTQA      | PerceptionTest&nbsp;(test) | STAR       | TVQA       | VideoMME        | TVBench      | ActivityNetQA   | EgoSchema&nbsp;(test) | TemporalBench    | TOMATO     | MotionBench&nbsp;(dev) | TempCompass&nbsp;(MCQ) | CGBench&nbsp;(clue) | Charades&nbsp;STA   | VideoHallucer   | Halluc.&nbsp;EventHallusion |
|:-------------:|:---------------------------:|:-----------------------:|:---------------------:|:-------------:|:-------------:|:--------------------------:|:----------:|:----------:|:----------------:|:-------------:|:--------------------:|:----------------------:|:---------------------:|:------------:|:------------------------:|:-----------------------:|:---------------------:|:-------------------:|:-------------------------------:|:--------------------------------:|
| PLM1B  | 92.5 | 34.3 | 86.4 | 70.1 | 80.3 | 72.7 | 83.7 | 50.3 | 49.2 | 50.4 | 62.5 | 60.4 | 18.2 | 25.5 | 52.2 | 64.6 | 43.6 | 55.2 | 49.2 | 79.5 |
| PLM3B  | 96.1 | 37.4 | 89.4 | 74.7 | 83.4 | 79.3 | 84.8 | 55.3 | 54.9 | 58.9 | 66.2 | 66.9 | 23.4 | 30.9 | 60.4 | 69.3 | 47.2 | 57.7 | 55.5 | 76.5 |
| PLM8B  | 99.7 | 35.9 | 90.7 | 77.1 | 84.1 | 82.7 | 84.9 | 59.3 | 58.3 | 63.5 | 67.3 | 68.8 | 28.3 | 33.2 | 61.4 | 72.7 | 46.4 | 58.6 | 57.7 | 77.3 |

See [`apps/plm/README.md`](apps/plm/README.md) for details and how to get started!

## Installation :wrench:
```shell
git clone https://github.com/facebookresearch/perception_models.git
cd perception_models

conda create --name perception_models python=3.12
conda activate perception_models

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu124

# We use torchcodec for decoding videos into PyTorch tensors
conda install ffmpeg -c conda-forge
pip install torchcodec==0.1 --index-url=https://download.pytorch.org/whl/cu124

pip install -e .
```
This will install an editable version of repo, allowing you to make changes to the code without needing to reinstall the package every time.


## 🙏 Acknowledgement
We are thankful to [Meta Lingua](https://github.com/facebookresearch/lingua) for releasing their code as open-source contributions. The code structure and code implementation of the LLM is directly forked from [Meta Lingua](https://github.com/facebookresearch/lingua). We are also thankful to [Open_CLIP](https://github.com/mlfoundations/open_clip) for open-source contributions in CLIP training, and [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) for CLIP model evaluation. 


## 📜 Citation
```BibTeX
@article{bolya2025PerceptionEncoder,
  title={Perception Encoder: The best visual embeddings are not at the output of the network},
  author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  journal={arXiv:2504.13181},
  year={2025}
}

@article{cho2025PerceptionLM,
  title={PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding},
  author={Jang Hyun Cho and Andrea Madotto and Effrosyni Mavroudi and Triantafyllos Afouras and Tushar Nagarajan and Muhammad Maaz and Yale Song and Tengyu Ma and Shuming Hu and Hanoona Rasheed and Peize Sun and Po-Yao Huang and Daniel Bolya and Suyog Jain and Miguel Martin and Huiyu Wang and Nikhila Ravi and Shashank Jain and Temmy Stark and Shane Moon and Babak Damavandi and Vivian Lee and Andrew Westbury and Salman Khan and Philipp Kr\"{a}henb\"{u}hl and Piotr Doll{\'a}r and Lorenzo Torresani and Kristen Grauman and Christoph Feichtenhofer},
  journal={arXiv:2504.13180},
  year={2025}
}
```
