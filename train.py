import os
import re
import yaml
import random
import argparse
import logging
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from s3prl.optimizers import get_optimizer

from utils import *
from models import init_model

from importlib import reload
logging.shutdown()
reload(logging)

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy

import wandb
from pytorch_lightning.loggers import WandbLogger


class W2V2Distil(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.yaml_cfg = cfg
        self.train_cfg = cfg['train']

        # Load teacher model
        teacher_model = self.yaml_cfg['teacher']['teacher_model']
        teacher_cfg = self.yaml_cfg['teacher']
        if 'wavlm' in teacher_model:
            self.teacher_model, teacher_config, self.task_agnostic = load_wavlm_and_config(teacher_model, arg_overrides=teacher_cfg)
        else:
            self.teacher_model, teacher_config, self.task_agnostic = load_model_and_config(teacher_model, arg_overrides=teacher_cfg)
        freeze_model(self.teacher_model)

        if self.train_cfg['mask_prob_schedule']:
            mask_prob_schedule = eval(self.train_cfg['mask_prob_schedule'])
            mask_prob_start, mask_prob_end = mask_prob_schedule[0], mask_prob_schedule[1]
            self.mask_prob_schedule = list(np.linspace(mask_prob_start, mask_prob_end, num=self.train_cfg['num_epochs']+1))
            self.teacher_model.model.mask_prob = self.mask_prob_schedule.pop(0)
            print("Teacher model mask_prob set to {:.4f}!".format(self.teacher_model.model.mask_prob))

        # Make student config independent of teacher
        self.distiller_cfg = self.yaml_cfg['distiller']
        init_student_config, init_student_model = init_model(self.yaml_cfg['model'])
        student_config = init_student_config(**self.distiller_cfg)
        student_config._teacher_task_agnostic = self.task_agnostic

        self.time_tag = get_time_tag()
        dump_yaml(student_config, self.yaml_cfg, self.time_tag)

        # Model Initialize -> Distillation training -> Add FC/Dropout & Fine-tuning
        self.student_model = init_student_model(
            cfg=student_config,
            teacher_model=self.teacher_model
        )

        # copy the mask tokens from the teacher
        # self.student_model.mask_emb.data = self.teacher_model.model.mask_emb.data
        if self.distiller_cfg['freeze_student_mask']:
            self.student_model.mask_emb.requires_grad = False

        self.rec_loss_weight = self.train_cfg['rec_loss_weight']
        self.rec_loss_type = self.train_cfg['rec_loss_type']
        self.random_layer_weight = self.train_cfg['random_layer_weight']

        if self.train_cfg['delete_projections']:
            self.student_model._disable_projection_heads()

        if self.train_cfg['specaug']:
            from utils.specaug import SpecAug
            specaug = SpecAug(**self.yaml_cfg['specaug'])
            self.student_model.add_specaug(specaug)

        if self.train_cfg['distil_random_layer'] > 0:
            self.num_encoders = self.distiller_cfg['encoder_layers']
            self.all_enc = range(self.num_encoders-1)
            self.rand_l = sorted(random.sample(self.all_enc, self.train_cfg['distil_random_layer']))
        else:
            assert self.train_cfg['random_layer_weight'] == 0

        self.batch_size = self.train_cfg['batch_size']
        self.num_gpus = self.train_cfg['gpus']
        if isinstance(self.num_gpus, list):
            self.num_gpus = len(self.num_gpus)
        data_cfg = self.yaml_cfg['data']
        bucketing_path = data_cfg['bucketing_path']
        libri_root = data_cfg['libri_root']
        train_set = data_cfg['train_set']
        test_set = data_cfg['test_set']

        # download & prepare data
        self.train_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=train_set,
            libri_root=libri_root,
        )
        self.eval_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=['dev-clean'],
            libri_root=libri_root,
        )
        self.test_data = LibriDataset(
            batch_size=self.batch_size,
            file_path=bucketing_path,
            sets=test_set,
            libri_root=libri_root,
        )

        # For better pytorch lightning logging
        logging.shutdown()
        reload(logging)

    def forward(self, x, padding_mask=None):
        # Seems like lightning had been using the teacher model as training mode the whole time
        self.teacher_model.eval()
        if self.distiller_cfg['update_teacher_mask']:
            self.teacher_model.model.mask_emb.data = self.student_model.mask_emb.data
        
        m_teacher_results = self.teacher_model.extract_features(
                            source=x.clone().contiguous(), 
                            padding_mask=padding_mask,
                            mask=True,
                            )
        # -> RETURNS: {
        #     "x": (B x T x D) (encoder output),
        #     "layer_results": [x, (attn, lr)] x #layers,
        #     "features": [features]
        #     "mask_indices": [B x T] (bool)
        # }

        unm_teacher_results = self.teacher_model.extract_features(
                              source=x.clone().contiguous(),
                              padding_mask=padding_mask,
                              mask=False,
                              )

        student_results = self.student_model(
            source=x.clone().contiguous(), 
            padding_mask=padding_mask,
            mask_indices=m_teacher_results['mask_indices'],
        )
        # -> RETURNS: {
        #     "x": x,
        #     "padding_mask": padding_mask,
        #     "layer_results": layer_results,
        #     "tr_layer_results": tr_layer_results,
        #     "projections": projections
        # }

        return student_results, unm_teacher_results, m_teacher_results 

    def forward_without_mask(self, x, padding_mask=None):
        # Seems like lightning had been using the teacher model as training mode the whole time
        self.teacher_model.eval()

        teacher_results = self.teacher_model.extract_features(
            source=x, 
            padding_mask=padding_mask,
            mask=False,
        )
        # -> RETURNS: {
        #     "x": (B x T x D) (encoder output),
        #     "layer_results": [x, (attn, lr)] x #layers,
        #     "features": [features]
        # }

        student_results = self.student_model(
            source=x, 
            padding_mask=padding_mask,
            mask_indices=None,
        )
        # -> RETURNS: {
        #     "x": x,
        #     "padding_mask": padding_mask,
        #     "features": features after post projector,
        #     "layer_results": layer_results,
        #     "tr_layer_results": tr_layer_results,
        #     "projections": projections
        # }

        return student_results, teacher_results

    def training_step(self, batch, batch_idx):
        student_results, unm_teacher_results, m_teacher_results = self(**batch)
        
        loss, losses = self.calculate_loss(student_results, unm_teacher_results, m_teacher_results)

        if self.train_cfg['monitor_losses']:
            for k, v in losses.items():
                self.log(k, v.item(), prog_bar=True)
            # mask_diff = F.mse_loss(self.teacher_model.model.mask_emb, self.student_model.mask_emb.detach(), reduction='mean')
            # self.log('mask_diff', mask_diff, prog_bar=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        if self.train_cfg['distil_random_layer'] > 0:
            self.rand_l = sorted(random.sample(self.all_enc, self.train_cfg['distil_random_layer']))
            # TODO: reset prog bar metrics
            # self.trainer._logger_connector.reset_metrics()
        if self.train_cfg['mask_prob_schedule']:
            try: 
                self.teacher_model.model.mask_prob = self.mask_prob_schedule.pop(0)
                if self.global_rank == 0:
                    print("Teacher model mask_prob set to {:.4f}!".format(self.teacher_model.model.mask_prob))
            except:
                pass
    
    def validation_step(self, batch, batch_idx):
        student_results, teacher_results = self.forward_without_mask(**batch)
        loss = self.calculate_loss_without_mask(student_results, teacher_results)
        self.log("v_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        return {"v_loss": loss}
    
    def test_step(self, batch, batch_idx):
        student_results, teacher_results = self.forward_without_mask(**batch)
        loss = self.calculate_loss_without_mask(student_results, teacher_results)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        return {"test_loss": loss}


    def calculate_loss(self, student_results, teacher_results, m_teacher_results, labels=None):
        losses = {}
        mask_indices = m_teacher_results['mask_indices'] # BxT

        # Feature loss
        if self.rec_loss_weight > 0:
            if self.train_cfg['distil_random_layer'] > 0:
                teacher_hiddens = [
                teacher_results["layer_results"][l][0].transpose(0, 1)
                    for l in self.rand_l
                ]
                teacher_hiddens.append(teacher_results["layer_results"][-1][0].transpose(0, 1))
                teacher_hiddens = torch.stack(teacher_hiddens, dim=1) # BxNxTxD
                
                m_teacher_hiddens = [
                m_teacher_results["layer_results"][l][0].transpose(0, 1)
                    for l in self.rand_l
                ]
                m_teacher_hiddens.append(m_teacher_results["layer_results"][-1][0].transpose(0, 1))
                m_teacher_hiddens = torch.stack(m_teacher_hiddens, dim=1) # BxNxTxD

                student_hiddens = [
                    student_results["projections"][l]
                    for l in self.rand_l
                ]
                student_hiddens.append(student_results["projections"][-1])
                pred = torch.stack(student_hiddens, dim=1)

            else:
                raise NotImplementedError

            target = teacher_hiddens.narrow(2, 0, pred.shape[2])
            m_target = m_teacher_hiddens.narrow(2, 0, pred.shape[2])
            B, N, T, D = target.shape
            mask_indices_ = mask_indices.unsqueeze(1).repeat_interleave(N-1, dim=1)

            if self.rec_loss_type == 'l1':
                mask_hint_loss = F.l1_loss(pred[:,:N-1][mask_indices_], target[:,:N-1][mask_indices_], reduction="none")
                mask_rec_loss = F.l1_loss(pred[:,-1][mask_indices], target[:,-1][mask_indices], reduction="none")
                unmask_hint_loss = F.l1_loss(pred[:,:N-1][~mask_indices_], m_target[:,:N-1][~mask_indices_], reduction="none")
                unmask_rec_loss = F.l1_loss(pred[:,-1][~mask_indices], m_target[:,-1][~mask_indices], reduction="none")
            elif self.rec_loss_type == 'mse':
                mask_hint_loss = F.mse_loss(pred[:,:N-1][mask_indices_], target[:,:N-1][mask_indices_], reduction="none")
                mask_rec_loss = F.mse_loss(pred[:,-1][mask_indices], target[:,-1][mask_indices], reduction="none")
                unmask_hint_loss = F.mse_loss(pred[:,:N-1][~mask_indices_], m_target[:,:N-1][~mask_indices_], reduction="none")
                unmask_rec_loss = F.mse_loss(pred[:,-1][~mask_indices], m_target[:,-1][~mask_indices], reduction="none")           
            else:
                raise NotImplementedError("rec_loss_type must be one of 'l1', 'mse'.")

            if self.train_cfg['distil_random_layer'] > 0:
                mask_hint_loss = mask_hint_loss.mean(-1).sum() / B / T
                unmask_hint_loss = unmask_hint_loss.mean(-1).sum() / B / T
                hint_loss = (mask_hint_loss + unmask_hint_loss) * self.random_layer_weight

                mask_rec_loss = mask_rec_loss.mean(-1).sum() / B / T
                unmask_rec_loss = unmask_rec_loss.mean(-1).sum() / B / T
                rec_loss = (mask_rec_loss + unmask_rec_loss) * self.rec_loss_weight
            else:
                raise NotImplementedError

        else:
            rec_loss = 0
            rec_layer_loss = 0
        
        losses['mask_hint_loss'] = mask_hint_loss
        losses['unmask_hint_loss'] = unmask_hint_loss
        losses['mask_rec_loss'] = mask_rec_loss
        losses['unmask_rec_loss'] = unmask_rec_loss
        losses['hint_loss'] = hint_loss
        losses['rec_loss'] = rec_loss
            
        total_loss = hint_loss + rec_loss
        
        return total_loss, losses

    def calculate_loss_without_mask(self, student_results, teacher_results, labels=None):
        losses = {}

        # Feature loss
        if self.rec_loss_weight > 0:
            if self.train_cfg['distil_random_layer'] > 0:
                teacher_hiddens = [
                teacher_results["layer_results"][l][0].transpose(0, 1)
                    for l in self.rand_l
                ]
                teacher_hiddens.append(teacher_results["layer_results"][-1][0].transpose(0, 1))
                teacher_hiddens = torch.stack(teacher_hiddens, dim=1) # BxNxTxD
                
                student_hiddens = [
                    student_results["projections"][l]
                    for l in self.rand_l
                ]
                student_hiddens.append(student_results["projections"][-1])
                pred = torch.stack(student_hiddens, dim=1)

            else:
                raise NotImplementedError

            target = teacher_hiddens.narrow(2, 0, pred.shape[2])

            if self.rec_loss_type == 'l1':
                rec_loss = F.l1_loss(pred, target, reduction="none")
            elif self.rec_loss_type == 'mse':
                rec_loss = F.mse_loss(pred, target, reduction="none")
            else:
                raise NotImplementedError("rec_loss_type must be one of 'l1', 'mse'.")

            if self.train_cfg['distil_random_layer'] > 0:
                rec_loss[:, :-1] = rec_loss[:, :-1] * self.random_layer_weight  # Hint-based distillation
                rec_loss[:, -1] = rec_loss[:, -1] * self.rec_loss_weight 
                rec_layer_loss = rec_loss.mean((0, 2, 3))
                rec_loss = rec_layer_loss.sum()
            else:
                with torch.no_grad():
                    rec_layer_loss = rec_loss.mean((0, 2, 3)) 
                rec_loss = rec_loss.mean()
        else:
            rec_loss = 0
        
        return rec_loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=eval(self.yaml_cfg['optimizer']['lr']))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.1, verbose=True)
        
        train_batches = len(self.train_dataloader()) // self.num_gpus
        num_training_steps = (self.train_cfg['num_epochs'] * train_batches) // self.train_cfg['accumulate_grad_batches']
        num_warmup_steps = int(num_training_steps * self.yaml_cfg['optimizer']['warmup_proportion'])

        return {
            "optimizer": get_optimizer(
                [self.student_model],
                num_training_steps,
                self.yaml_cfg['optimizer']
            )
        }

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=1,
                          shuffle=True,
                          collate_fn=self.train_data.collate_fn,
                          num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.eval_data,
                          batch_size=1,
                          collate_fn=self.eval_data.collate_fn,
                          num_workers=16)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=1,
                          collate_fn=self.test_data.collate_fn,
                          num_workers=16)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '-cfg', '--config', 
                        help='yaml config path for training')

    parser.add_argument('-m', '--model', default='armhubert',
                        help='define model name')

    parser.add_argument('-t', '--test',
                        action='store_true', help='Enable testing mode')

    args = parser.parse_args()

    YAML_PATH = args.config or './conf/armhubert/armhubert-960.yaml'
    with open(YAML_PATH) as f:
        YAML_CFG = yaml.load(f, Loader = yaml.FullLoader)

    YAML_CFG['model'] = args.model

    batch_size = YAML_CFG['train']['batch_size']
    output_dir = YAML_CFG['train']['base_dir'] + 'results/pretrain/' + YAML_CFG['train']['output_dir']
    checkpoint = YAML_CFG['train']['checkpoint']
    gpus = YAML_CFG['train']['gpus']
    num_epochs = YAML_CFG['train']['num_epochs']
    use_fp16 = 16 if YAML_CFG['train']['use_fp16'] else 32
    use_apex = 'apex' if YAML_CFG['train']['use_apex'] else 'native'
    accumulate_grad_batches = YAML_CFG['train']['accumulate_grad_batches']

    model = W2V2Distil(cfg = YAML_CFG)
    wandb_logger = WandbLogger(project = 'ARMHuBERT',
                               name = model.time_tag,
                               resume = False,
                               sync_tensorboard = True)


    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='checkpoint-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='v_loss',
        mode='min'
    )

    # early_stopping = EarlyStopping(
    #     monitor='v_loss',
    #     patience=15,
    #     verbose=True,
    #     mode='min'
    # )

    trainer = Trainer(
        accelerator = 'gpu',
        devices = 1 if args.test else -1,
        strategy= DDPStrategy(find_unused_parameters=False),
        amp_backend=use_apex,
        #amp_backend = "apex",
        #amp_level = "O2",
        precision=use_fp16, 
        max_epochs=num_epochs,
        sync_batchnorm=True,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=checkpoint_callback,  # [early_stopping, checkpoint_callback]
        logger = wandb_logger,
    )

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(
            model, 
            ckpt_path=os.path.join(output_dir, checkpoint) if checkpoint else None
        )

 
