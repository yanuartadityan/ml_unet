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

Testing in few epochs:

```
100%|████████████████████████████████████████████████████████████████████| 159/159 [02:22<00:00,  1.12it/s, loss=0.199]
::> saving checkpoint...
::Got 1822750/1843200 with acc 0.9889051914215088
::Dice score is 0.9708409309387207
100%|████████████████████████████████████████████████████████████████████| 159/159 [01:48<00:00,  1.46it/s, loss=0.149]
::> saving checkpoint...
::Got 1830536/1843200 with acc 0.9931293725967407
::Dice score is 0.9814187288284302
100%|████████████████████████████████████████████████████████████████████| 159/159 [01:46<00:00,  1.49it/s, loss=0.119]
::> saving checkpoint...
::Got 1832885/1843200 with acc 0.9944037795066833
::Dice score is 0.984639585018158
```