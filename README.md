# self-training-self-supervised-disfluency

Combining Self-Training and Self-Supervised Learning for Unsupervised
Disfluency Detection

This repo contains the code and model used for Combining Self-Training and Self-Supervised Learning for Unsupervised
Disfluency Detection (EMNLP 2020).

All the code and model are released. Thank you for your patience!

## About Model

We release our self-supervised model trained by pseudo data and grammar check model. Please download it in the following link, and put model in "self_supervised_model" and "grammar_check_model " folder.

[grammar_check_model][grammar_check_model]

[self_supervised_model][self_supervised_model]


[self_supervised_model]:https://drive.google.com/file/d/1MQ-uJW6HSsvLDuF4IUFl81lGRGQXUrgr/view?usp=sharing

[grammar_check_model]:https://drive.google.com/file/d/1nlWvMJm54MJ_HsA315CEiSnBGDclXn92/view?usp=sharing

## How to Use

```
conda create -n ss_disfluency python=3.7
conda activate ss_disfluency
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
cd transformers
python setup.py install
cd ..
nohup sh run.sh 0 > log_run 2>&1 &
```
## About GPU

This repo need to be run on GPU, and it will cost 10G GPU RAM.

## Citation

```
@article{wang2020combining,
  title={Combining Self-Training and Self-Supervised Learning for Unsupervised Disfluency Detection},
  author={Wang, Shaolei and Wang, Zhongyuan and Che, Wanxiang and Liu, Ting},
  journal={arXiv preprint arXiv:2010.15360},
  year={2020}
}
```

## Contact
zywang@ir.hit.edu.cn and slwang@ir.hit.edu.cn
