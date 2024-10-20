import torch
from torch import optim
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
from utils.compute_scores import get_metrics, get_four_metrics
import numpy as np


class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self, epoch):
        raise NotImplementedError()

    def test(self, epoch):
        raise NotImplementedError()


class MSDTrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, test_loader, model, args, logger):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.args = args
        self.step = 0
        self.logger = logger
        self.refresh_step = 2
        self.best_val_acc, self.best_val_f1, self.best_val_loss = 0.0, 0.0, np.Inf
        self.best_epoch = None
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.global_loss_weight = args.global_loss_weight
        self.token_loss_weight = args.token_loss_weight
        self.ce_loss_weight = args.ce_loss_weight
        if self.train_loader is not None:
            self.train_num_steps = len(self.train_loader) * args.num_epochs

    def train(self):
        self.before_train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_loader) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Pretrained-model learning rate = {}".format(self.args.pretrained_lr))
        self.logger.info("  Other learning rate = {}".format(self.args.other_lr))
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            for epoch in range(1, self.args.num_epochs+1):
                self.model.train()
                train_loss, total = 0.0, 0.0
                train_avg_loss = 0.0
                predict, real_label = [], []
                pbar.set_description_str(desc='Epoch {}/{}'.format(epoch, self.args.num_epochs))
                for batch_idx, batch in enumerate(self.train_loader):
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    if self.args.clip_knowledge:
                        input_ids, attention_mask, batch_imgs, batch_labels = batch
                    else:
                        input_ids, token_type_ids, attention_mask, batch_imgs, batch_labels = batch
                    
                    batch_len = len(batch_imgs)

                    if epoch > 1:
                        alpha = self.args.alpha
                    else:
                        alpha = self.args.alpha * min(1, batch_idx / len(self.train_loader))

                    with torch.set_grad_enabled(True):
                        if self.args.clip_knowledge:
                            output, loss_itc, loss_md = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                images=batch_imgs,
                                alpha=alpha,
                                mode='train'
                            )
                        else:
                            output, loss_itc, loss_md = self.model(
                                input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                images=batch_imgs,
                                alpha=alpha,
                                mode='train'
                            )
                        
                        loss = self.ce_loss(output, batch_labels.to(self.args.device))
                        total_loss = self.ce_loss_weight * loss \
                                + self.global_loss_weight * loss_itc \
                                + self.token_loss_weight * loss_md

                        total_loss = (total_loss - 2.0).abs() + 2.0
                        total_loss.backward()
                        train_avg_loss += float(total_loss.detach().item())
                        train_loss += float(total_loss.detach().item())
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                    predict = predict + get_metrics(output.cpu())
                    real_label = real_label + batch_labels.cpu().numpy().tolist()
                    total += batch_len

                    if self.step % self.refresh_step == 0:
                        train_avg_loss = float(train_avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(train_avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        train_avg_loss = 0.0

                train_acc, train_recall, train_pre, train_f1 = get_four_metrics(real_label, predict)
                print('\n', '-='*50)
                self.logger.info("***** Train Eval results *****")
                self.logger.info('Train Epoch: {} Loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(
                    epoch, train_loss / len(self.train_loader), train_acc, train_recall, train_pre, train_f1))

                _, _, avg_val_f1 = self.evaluate(epoch=epoch)
                _, _, avg_cur_f1 = self.test(epoch=epoch)
                if avg_val_f1 >= self.best_val_f1:
                    self.best_val_f1 = avg_val_f1
                    self.best_epoch = epoch
                    if self.args.save_model_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_model_path+"/best_model.pth")
                #         self.logger.info("Epoch {} Save best model at {}".format(epoch, self.args.save_model_path))
                # self.logger.info("Get better performance at epoch {}".format(self.best_epoch))
            pbar = None

    def evaluate(self, epoch):
        val_loss = 0.0
        predict, real_label = [], []
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc='Val')
                for batch_idx, batch in enumerate(self.val_loader):
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    if self.args.clip_knowledge:
                        input_ids, attention_mask, batch_imgs, batch_labels = batch
                        output, loss_md = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            images=batch_imgs,
                            alpha=self.args.alpha,
                            mode='dev'
                        )
                    else:
                        input_ids, token_type_ids, attention_mask, batch_imgs, batch_labels = batch
                        output, loss_md = self.model(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            images=batch_imgs,
                            alpha=self.args.alpha,
                            mode='dev'
                        )

                    loss = self.ce_loss(output, batch_labels.to(self.args.device))
                    total_loss = loss + loss_md
                    val_loss += float(total_loss.clone().detach().item())
                    predict = predict + get_metrics(output.cpu())
                    real_label = real_label + batch_labels.cpu().numpy().tolist()
                    pbar.update()
                pbar.close()
        val_acc, val_recall, val_pre, val_f1 = get_four_metrics(real_label, predict)
        self.logger.info("***** Dev Eval results *****")
        self.logger.info('Val Epoch: {} Loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(
            epoch, val_loss / len(self.val_loader), val_acc, val_recall, val_pre, val_f1))
        return val_acc, val_loss, val_f1

    def test(self, epoch=None):
        test_loss = 0.0
        predict, real_label = [], []
        self.model.eval()
        if self.args.load_model_path is not None:
            self.logger.info("Loading model from {}".format(self.args.load_model_path))
            self.model.load_state_dict(torch.load(self.args.load_model_path))
            self.logger.info("Load model successful!")
        with torch.no_grad():
            with tqdm(total=len(self.test_loader), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc='Test')
                for batch_idx, batch in enumerate(self.test_loader):
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)

                    if self.args.clip_knowledge:
                        input_ids, attention_mask, batch_imgs, batch_labels = batch
                        output, loss_md = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            images=batch_imgs,
                            alpha=self.args.alpha,
                            mode='test'
                        )
                    else:
                        input_ids, token_type_ids, attention_mask, batch_imgs, batch_labels = batch
                        output, loss_md = self.model(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            images=batch_imgs,
                            alpha=self.args.alpha,
                            mode='test'
                        )
                    loss = self.ce_loss(output, batch_labels.to(self.args.device))
                    total_loss = loss + loss_md
                    test_loss += float(total_loss.clone().detach().item())
                    predict = predict + get_metrics(output.cpu())
                    real_label = real_label + batch_labels.cpu().numpy().tolist()
                    pbar.update()
                pbar.close()
        test_acc, test_recall, test_precision, test_f1 = get_four_metrics(real_label, predict)
        # if self.args.load_model_path is not None:
        self.logger.info("***** Test Eval results *****")
        self.logger.info('Test: Loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(
            test_loss / len(self.test_loader), test_acc, test_recall, test_precision, test_f1))
        return test_acc, test_loss, test_f1

    def before_train(self):
        no_decay = ['bias', 'LayerNorm']
        if self.args.clip_knowledge:
            parameters = [
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    not any(nd in name for nd in no_decay) and (
                            'text2vit_atten' in name or
                            'vit2text_atten' in name or
                            'dict_atten' in name
                    )],
                 'weight_decay': 1e-2,
                 'lr': self.args.pretrained_lr},
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    any(nd in name for nd in no_decay) and (
                            'text2vit_atten' in name or
                            'vit2text_atten' in name or
                            'dict_atten' in name
                    )],
                 'weight_decay': 0.0,
                 'lr': self.args.pretrained_lr},
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    not any(nd in name for nd in no_decay) and ('multimodal_clip_encoder' in name)],
                 'weight_decay': 1e-2,
                 'lr': self.args.clip_lr},
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    any(nd in name for nd in no_decay) and ('multimodal_clip_encoder' in name)],
                 'weight_decay': 0.0,
                 'lr': self.args.clip_lr},
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    not any(nd in name for nd in no_decay) and (
                            'first_classifier' in name or
                            'second_classifier' in name or
                            'vit2vit_linear' in name or
                            'vision_linear' in name or
                            'text_linear' in name or
                            'dict_feature' in name or
                            'text_dict_cls' in name or
                            'vision_dict_cls' in name
                    )],
                 'weight_decay': 1e-2,
                 'lr': self.args.other_lr},
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    any(nd in name for nd in no_decay) and (
                            'first_classifier' in name or
                            'second_classifier' in name or
                            'vit2vit_linear' in name or
                            'vision_linear' in name or
                            'text_linear' in name or
                            'dict_feature' in name or
                            'text_dict_cls' in name or
                            'vision_dict_cls' in name
                    )],
                 'weight_decay': 0.0,
                 'lr': self.args.other_lr},
            ]

        else:
            parameters = [
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    not any(nd in name for nd in no_decay) and (
                            'text_encoder' in name or
                            'vision_encoder' in name or
                            'text2vit_atten' in name or
                            'vit2text_atten' in name or
                            'dict_atten' in name
                    )],
                 'weight_decay': 1e-2,
                 'lr': self.args.pretrained_lr},
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    any(nd in name for nd in no_decay) and (
                            'text_encoder' in name or
                            'vision_encoder' in name or
                            'text2vit_atten' in name or
                            'vit2text_atten' in name or
                            'dict_atten' in name
                    )],
                 'weight_decay': 0.0,
                 'lr': self.args.pretrained_lr},
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    not any(nd in name for nd in no_decay) and (
                            'first_classifier' in name or
                            'second_classifier' in name or
                            'vision_proj' in name or
                            'text_proj' in name or
                            'dict_feature' in name or
                            'text_dict_cls' in name or
                            'vision_dict_cls' in name
                    )],
                 'weight_decay': 1e-2,
                 'lr': self.args.other_lr},
                {'params': [
                    p for name, p in self.model.named_parameters() if
                    any(nd in name for nd in no_decay) and (
                            'first_classifier' in name or
                            'second_classifier' in name or
                            'vision_proj' in name or
                            'text_proj' in name or
                            'dict_feature' in name or
                            'text_dict_cls' in name or
                            'vision_dict_cls' in name
                    )],
                 'weight_decay': 0.0,
                 'lr': self.args.other_lr},
            ]

        self.optimizer = optim.AdamW(parameters)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * len(self.train_loader) * self.args.warmup_epochs,
            num_training_steps=len(self.train_loader) * self.args.warmup_epochs
        )
