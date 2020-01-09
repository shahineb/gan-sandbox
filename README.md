# Play around with GANs

_PyTorch implementation of [Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets) and play around with [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), github-like [Identicon](https://en.wikipedia.org/wiki/Identicon) images and [Pokemon sprints](https://www.dropbox.com/s/860hi05kyl0yxaa/pokemon.zip?dl=0)._

<p align="center">
<img width="60%" src="https://github.com/shahineb/gan-sandbox/blob/master/docs/img/identicon_gans.jpeg" />
</p>
<p align="center">
<em> Intermediate Results of Identicon Generation </em>
</p>  

## Repository structure

```
├── bin
│   ├── train.py
│   ├── dashboard.ipynb
│   ├── session_1
│   ├── session_2
│   ├── ...
│   └── session_n
├── core
│   ├── dataloader
│   │   ├── transforms
│   │   └── utils
│   ├── engine
│   │   ├── config_file.py
│   │   ├── trainer.py
│   │   └── utils
│   └── models
│       ├── backbones
│       └── modules
├── data
├── docs
└── utils
```

- `data`: contains actual data i.e. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) faces and [Pokemon sprints](https://www.dropbox.com/s/860hi05kyl0yxaa/pokemon.zip?dl=0), structured according to provider conventions
- `docs`: any paper, notes, figures relevant to this repository
- `bin`: `dashboard.ipynb` is our UI to setup experience sessions, setting up an associated directory `session_i`. The experiment is then launched through executing `train.py`, [here](https://github.com/shahineb/gan-sandbox/tree/master/bin) for more details
- `core`: contains definition of data processing and loading protocols, models and training engines


# References
```
@incollection{NIPS2014_5423,
title = {Generative Adversarial Nets},
author = {Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
booktitle = {Advances in Neural Information Processing Systems 27},
editor = {Z. Ghahramani and M. Welling and C. Cortes and N. D. Lawrence and K. Q. Weinberger},
pages = {2672--2680},
year = {2014},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf}
}
```
