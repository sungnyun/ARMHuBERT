# Recycle-and-Distill (Interspeech 2023)
[arXiv](https://arxiv.org/abs/2305.11685) | [BibTeX](#bibtex)

![ARMHuBERT-figure](https://github.com/sausage-333/SSL-Distillation/assets/54805233/2fca535e-014e-40c0-81b9-6d7dd22fc19a)

[**Recycle-and-Distill: Universal Compression Strategy for Transformer-based
Speech SSL Models with Attention Map Reusing and Masking Distillation**]("~"), INTERSPEECH 2023.

[Kangwook Jang](https://github.com/sausage-333)\*,
[Sungnyun Kim](https://github.com/sungnyun)\*,
[Se-Young Yun](https://fbsqkd.github.io), [Hoirin Kim](https://scholar.google.com/citations?user=naLHjOsAAAAJ&hl=en)<br/>
\* equal contribution

- **Attention Map Reusing**: Reuse previous layer's attention map to remove key & query parameters in Transformer
- **Masking Distillation**: Masking distillation treating masked frames and unmasked frames separately
- Parameters and MACs of ARMHuBERT have decreased to **28% and 30%** of the teacher, HuBERT Base, respectively.
- ARMHuBERT achieves **PER of 7.72%, WER of 9.96%** on the SUPERB benchmark in an E2E distillation manner.


## Requirements
Install the necessary packages with: 
```
$ pip install -r requirements.txt
```


## Distillation
1. Download the teacher model checkpoint to perform knowledge distillation, and place it under the root path, `./`.

	+ For HuBERT Base: [link](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) (`hubert_base_ls960.pt`)
	+ For wavLM Base: [link](https://huggingface.co/s3prl/converted_ckpts/tree/main) (`wavlm_base.pt`)

2. Download the [LibriSpeech](https://www.openslr.org/12) dataset.

	+ For 100h distillation, download `train-clean-100`
	+ For 960h distillation, download whole dataset, `train-clean-100`, `train-clean-360`, `train-other-500`
	+ For validation, download `dev-clean`
		+ You can validate your model with test clean other either. In this case, please download `test-clean`, and modify `self.eval_data` in `train.py` file.

3. Modify the configuration file in `./conf/[model_name]/[config].yaml`.    
	+ For example, the configuration file `./conf/armhubert/armhubert-960.yaml` contains all the settings for reproducing ARMHuBERT on LibriSpeech 960h dataset.	
	+ Set the path to the teacher model checkpoint at `teacher_model`, and the root path to the LibriSpeech dataset at `libri_root`. 

4. Then, run the following command:
```
python train.py -c ./conf/[model_name]/[config].yaml
```

For ARMHuBERT,
	```
	python train.py -c ./conf/armhubert/armhubert-960.yaml
	```

After training, the model checkpoints and the corresponding configuration file will be created at `./results/pretrain/`.


## Fine-tuning
1. Clone and install the [S3PRL toolkit](https://github.com/s3prl/s3prl) with ```pip install -e ".[all]"``` (dev mode).

2. Copy the entire `./models/[model_name]` folder into `<s3prl root>/s3prl/upstream/`.

3. Please add upstream importing line in `<s3prl root>/s3prl/hub.py`.
	
	```
	from s3prl.upstream.[model_name].hubconf import *
	```
	For ARMHuBERT,
	```
	from s3prl.upstream.armhubert.hubconf import *
	```

4. Please change each config file of s3prl downstream tasks as follows.
	+ Uncomment learning rate scheduler
	+ Learning rate scaled to 10x in spekaer identification (SID) task

5. Run the following command to fine-tune the ARMHuBERT model.

	For automatic speech recognition (ASR) as an example:
	```
	python run_downstream.py \
	-m train \
	-n ARMHuBERT-ASR \  # You can set your exp name whatever you want
	-u armhubert \
	-d asr \
	-k <path to .ckpt file in <git root>/results/pretrain/> \
	-g <path to .yaml file in <git root>/results/pretrain/>
	```
	Note: Refer to the [SUPERB docs](https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/docs/superb.md) for more information on usage details and data preparation.

## Result

![ARMHuBERT-results](https://github.com/sausage-333/SSL-Distillation/assets/54805233/5c76394c-82d0-4515-afd1-8dd4c022a793)

We evaluate our student models on the SUPERB benchmark.

MaskHuBERT highly improves the performances in content- and semantics-related tasks. See PR, ASR, SF, and IC.

ARMHuBERT shows promising improvements when compared to MaskHuBERT in SF and SID tasks, exhibiting a similar level of performance in other tasks.

ARMHuBERT achieves a better overall score of **78.1** with less parameters than MaskHuBERT.
This is an state-of-the-art performance for an end-to-end distillation approach such as [Deep-versus-wide 12-L](https://arxiv.org/abs/2207.06867?context=eess.AS) or [FitHuBERT](https://arxiv.org/abs/2207.00555).

You can also check that our model works on other Transformer backbone model, [wavLM](https://arxiv.org/abs/2110.13900), too.

## Try this distillation strategy with your Transformer backbone models
We have only performed evaluation on HuBERT-based models, but this strategy can be performed identically on any speech model with a Transformer backbone. E.g. [AST](https://arxiv.org/abs/2104.01778) (Audio Spectrogram Transformer).


## Checkpoint
For our checkpoints, check this [link](https://kaistackr-my.sharepoint.com/:f:/g/personal/dnrrkdwkd12_kaist_ac_kr/EpL0YsQg__RJlmsq__Y4UZIBtXuORYXLyAhUX3bK69e8DA?e=EAIEWP)!

## BibTeX
If you find this repo useful for your research, please consider citing our paper:
```
@article{jang2023recycleanddistill,
         title={Recycle-and-Distill: Universal Compression Strategy for Transformer-based Speech SSL Models with Attention Map Reusing and Masking Distillation}, 
         author={Kangwook Jang and Sungnyun Kim and Se-Young Yun and Hoirin Kim},
	 journal={arXiv preprint arXiv:2305.11685},
         year={2023}
}
```

## Contact
- Kangwook Jang: dnrrkdwkd12@kaist.ac.kr
- Sungnyun Kim: ksn4397@kaist.ac.kr
