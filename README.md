# GIB: Gated Information Bottleneck for Generalization in Sequential Environments

Pytorch implmentation of the Gated Infromation Bottleneck (GIB) for Generalization in Sequential Environments (https://arxiv.org/abs/2110.06057)


## Abstract ##
Deep neural networks suffer from poor generalization to unseen environments when the underlying data distribution is different from that in the training set. By learning minimum sufficient representations from training data, the information bottleneck (IB) approach has demonstrated its effectiveness to improve generalization in different AI applications. In this work, we propose a new neural network-based IB approach, termed gated information bottleneck (GIB), that dynamically drops spurious correlations and progressively selects the most task-relevant features across different environments by a trainable soft mask (on raw features). GIB enjoys a simple and tractable objective, without any variational approximation or distributional assumption. We empirically demonstrate the superiority of GIB over other popular neural network-based IB approaches in adversarial robustness and out-of-distribution (OOD) detection. Meanwhile, we also establish the connection between IB theory and invariant causal representation learning, and observed that GIB demonstrates appealing performance when different environments arrive sequentially, a more practical scenario where invariant risk minimization (IRM) fails.


![Illustration of the Deterministic Gate](https://github.com/falesiani/git_public_tmp/blob/main/GIB_illustrative.png)



[Slides](https://github.com/falesiani/git_public_tmp/blob/main/ICDM_Presentation-2021.pdf) of the presentation at ICDM 2021


## Install Instructions
```console
pip install GIB
```

<!-- 
## Requirements: 
### Pytorchn:
* numpy==1.18.5
* matplotlib==3.1.1
* scikit-learn==0.24.1
 -->


## Reference

```bibtex
@inproceedings{alesiani2021gated,
  title={Gated Information Bottleneck for Generalization in Sequential Environments},
  author={Alesiani, Francesco and Yu, Shujian and Yu, Xi},
  booktitle={2021 IEEE International Conference on Data Mining (ICDM)},
  pages={1--10},
  year={2021},
  organization={IEEE}
}
```
