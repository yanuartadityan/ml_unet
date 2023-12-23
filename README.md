### UNET implementation with PyTorch from Scratch

References that are used:

* [Paper](https://arxiv.org/abs/1505.04597)
* [Coding](https://www.youtube.com/watch?v=IHq1t7NxS8k&t=221s)

Dataset:

Anything, but so far only tested on RGB (3-channels). Model should support 1 channel as well, but if you want it as default make sure to change `model.py`

```python
  class UNet(nn.Module):
    ...
    def __init__(self, in_channels=1, ...) 
    ...
```