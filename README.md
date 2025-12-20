# quantization
This project aims to understand the impact of  the quantization of image generation models on the performance of common fake image detectors.

# usage
- run `hf auth <your_hugging_face_token>` to be able to use the models
- if you need to choose a different GPU you can do `CUDA_VISIBLE_DEVICES=1 `

for txt2img run:
```
python txt2img.py --prompt "photo of an english setter" --output_dir /media/SSD_4TB/crispy_storage/
```

for now see README_1 for the rest
