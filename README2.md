

## Installation
This work is built from the [IP-Adapter](https://ip-adapter.github.io/). Please follow the following instructions to get IP-Adapter for Stable Diffusion XL ready.

Install IP Adaptor and download the needed models:
```
# install ip-adapter
git clone https://github.com/tencent-ailab/IP-Adapter.git
mv IP-Adapter/ip_adapter ip_adapter
rm -r IP-Adapter/
```

## Download Models

You can download models from [here](https://huggingface.co/h94/IP-Adapter) and store it by running:

```
# download the models
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
```
