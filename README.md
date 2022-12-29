# Variational Neural Cellular Automata

Code for the paper [Variational Neural Cellular Automata](https://arxiv.org/abs/2201.12360).

![VNCA generating faces](celeb.gif)

![VNCA re-generating digits](mnist.gif)


## Cite
```
@inproceedings{palm2022variational,
  title={Variational Neural Cellular Automata},
  author={Palm, Rasmus Berg and Gonz{\'a}lez-Duque, Miguel and Sudhakaran, Shyam and Risi, Sebastian},
  booktitle = {10th International Conference on Learning Representations, {ICLR} 2022},
  year={2022}
}
```

## Experiments

The following git revisions corresponds to the experiments/figured as listed.

| Experiment  | Figure(s) | Revision |
| --- | --- | --- |
| Doubling MNIST | 3, 4 | `036578c` |
| Latent space exploration | 5 | `a98bd68` |
| Doubling CelebA | 6 | `641588a` |
| Doubling CelebA w. beta=100 | 7, 8 | `8d1ee6e` | 
| MNIST damage | 9 | `4b2dc09` |
| CelebA damage | 10 | `a17aafd` |


## License

MIT License

Copyright (c) 2022 Rasmus Berg Palm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
