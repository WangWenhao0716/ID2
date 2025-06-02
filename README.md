# ID2
[ICML 2025] The official implementation of "[Origin Identification for Text-Guided Image-to-Image Diffusion Models](https://arxiv.org/abs/2501.02376)".

<p align="center">
  <img src="https://github.com/WangWenhao0716/ID2/blob/main/method_icml.png" width="70%">
</p>


## Dataset
We release the dataset at [HuggingFace](https://huggingface.co/datasets/WenhaoWang/OriPID). Please follow the instructions here to download the OriPID dataset, including training, query, and reference images.
The training code below uses VAE-encoded training images; for your convenience, you can also download these features directly.

## Demonstration
<p align="center">
  <img src="https://github.com/WangWenhao0716/ID2/blob/main/true.png" width="100%">
</p>

## Installation
```
conda create -n id2 python=3.9
conda activate id2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate
pip install timm==0.4.12
git clone https://github.com/WangWenhao0716/ID2.git
cd ID2
```
## Training

## Feature Extraction

### Step 1: Get VAE embedding.
```python
from diffusers import AutoPipelineForImage2Image
import torchvision
import torch
from PIL import Image
import requests

pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float32, variant="fp16", use_safetensors=True)
vae = pipeline.vae

mean, std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
transforms = torchvision.transforms.Compose([
  torchvision.transforms.Resize((256, 256)),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize([0.5], [0.5]), 
])

url = "https://huggingface.co/datasets/WenhaoWang/AnyPattern/resolve/main/Irises.jpg"
image = Image.open(requests.get(url, stream=True).raw)
latents = vae.encode(transforms(image).unsqueeze(0)).latent_dist.sample() # torch.Size([1, 4, 32, 32])
features = latents.reshape(len(latents), -1) # torch.Size([1, 4096])
```
### Step 2: Perform linear transformation.

If you train a model by yourself, the weight is obtained by:
```python
import torch
mod = torch.load('logs/sd2_d_multi/vit_sd2_vae_onelayerw_512f/checkpoint_24_ema.pth.tar', map_location='cpu')
torch.save(mod['state_dict']['module.base.0.fc1.weight'], 'vit_sd2_vae_onelayerw_512f.pth.tar')

```
Or you can directly download our trained model by:
```
wget https://github.com/WangWenhao0716/ID2/raw/refs/heads/main/vit_sd2_vae_onelayerw_512f.pth.tar
```

Then:
```python
import torch
W = torch.load('vit_sd2_vae_onelayerw_512f.pth.tar', map_location='cpu')
features_final = features@W.T # torch.Size([1, 4096]) -> torch.Size([1, 512])
```

The `features_final` is used for matching, i.e., comparing cosine similarity.

## Citation
```
@article{wang2025origin,
  title={Origin Identification for Text-Guided Image-to-Image Diffusion Models},
  author={Wang, Wenhao and Sun, Yifan and Yang, Zongxin and Tan, Zhentao and Hu, Zhengdong and Yang, Yi},
  journal={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=46n3izUNiv}
}
```

## License

We release the code and trained models under the [CC-BY-NC-4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). 

## Contact

If you have any questions or want to use this project commerically, feel free to contact [Wenhao Wang](https://wangwenhao0716.github.io/) (wangwenhao0716@gmail.com).
