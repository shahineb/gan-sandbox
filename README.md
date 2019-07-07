# Neural Conditioner

PyTorch implementation of Neural Conditioner introduced in [_"Learning about an exponential amount of conditional distributions"_](https://arxiv.org/abs/1902.08401)

# TODO

- [ ] Create Logger > Trainer attribute => modify fit and any other places logger should occur
- [x] Make VAE implementation more general and prone to experiments
- [ ] Add network initializer
- [x] Feature to compute hidden dimension numel = f(pooling, strides, padding, kernel_size)
- [x] Make Discriminator implementation more general and prone to experiments
- [x] Checkout AverageMeter vs Logger pluggin
- [ ] Checkout adversarial training scripts to see how adversarial balance is handled

# References
```
@article{DBLP:journals/corr/abs-1902-08401,
  author    = {Mohamed Ishmael Belghazi and
               Maxime Oquab and
               Yann LeCun and
               David Lopez{-}Paz},
  title     = {Learning about an exponential amount of conditional distributions},
  journal   = {CoRR},
  volume    = {abs/1902.08401},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.08401},
  archivePrefix = {arXiv},
  eprint    = {1902.08401},
  timestamp = {Tue, 21 May 2019 18:03:36 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1902-08401},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
