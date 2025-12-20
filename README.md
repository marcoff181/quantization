# quantization
This project aims to understand the impact of  the quantization of image generation models on the performance of common fake image detectors.

# usage
- run `hf auth login` and insert your hugging face token when prompted to be able to use the models
- if GPU 0 is busy (check with `nvidia-smi`) you can run on GPU 1 with `CUDA_VISIBLE_DEVICES=1 `

for txt2img run with 10 of the default prompts:
```
python txt2img.py --prompts 10 --output_dir /media/SSD_4TB/crispy_storage/
```

for now see README_1 for the rest
