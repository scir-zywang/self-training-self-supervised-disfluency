# self-training-self-supervised-disfluency

Combining Self-Training and Self-Supervised Learning for Unsupervised
Disfluency Detection

This repo contains the code and model used for Combining Self-Training and Self-Supervised Learning for Unsupervised
Disfluency Detection (EMNLP 2020).

All the code and model are released. Thank you for your patience!

## About Model

We release our self-supervised model trained by pseudo data and grammar check model. Please download it in the following link, and put model in "self_supervised_model" and "grammar_check_model " folder.

(still be uploading)

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

## Contact
zywang@ir.hit.edu.cn and slwang@ir.hit.edu.cn
