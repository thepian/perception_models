# Zero-Shot ClipBench Evaluation
Please download the supported datasets and update paths clip_benchmark/datasets/. And run
```bash
model='PEv1-G14-448'
CHECKPOINT='PATH_TO_PE_Core_G14_448'
DATASETS=./clip_benchmark/tasks/wds_benchmarks.txt

python -m clip_benchmark.cli eval \
    --model $model \
    --pretrained $CHECKPOINT \
    --dataset "$DATASETS" \
    --dataset_root "/checkpoint/vision_encoder/dataset01/benchmark/{dataset_cleaned}/" \
    --output "./benchmark_{pretrained}_{dataset}_{num_frames}_{model}_{language}_{task}.json" \
    --force-preprocess-cfg resize_mode=squash

```
This script will perform zero-shot classification or retireval benchmarks defined in clip_benchmark/tasks/wds_benchmarks.txt. Examples above includes the following tasks:
- ImageNet 1K classification
- ImageNet v2 classification
- ImageNet Adversial classification
- MS-COCO retrieval
- Flickr30K retrieval