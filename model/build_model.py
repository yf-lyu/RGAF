import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel, CLIPModel
from copy import deepcopy


class Attention(nn.Module):
    def __init__(self, num_attention_head, hidden_size, dropout_prob) -> None:
        super().__init__()
        self.num_attention_head = num_attention_head
        self.hidden_size = hidden_size
        self.attention_head_size = int(self.hidden_size / self.num_attention_head)
        self.query = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.key = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.value = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask=None, encoder_attention_mask=None, output_attention=False):
        query_layer = self.transpose_for_scores(self.query(query))
        key_layer = self.transpose_for_scores(self.key(key))
        value_layer = self.transpose_for_scores(self.value(value))

        if encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attention else (context_layer, )
        return outputs


class LayerNorm(nn.Module):
    def __init__(self, x_size, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.ones_tensor = nn.Parameter(torch.ones(x_size))
        self.zeros_tensor = nn.Parameter(torch.zeros(x_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.ones_tensor * (x - mean) / (std + self.eps) + self.zeros_tensor


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout, layer_norm_eps) -> None:
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size
        )
        self.w_2 = nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size
        )
        self.layer_norm = nn.LayerNorm(
            normalized_shape=hidden_size,
            eps=layer_norm_eps
        )
        self.dropout_1 = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        return self.dropout_2(self.w_2(inter))


class SublayerConnecttion(nn.Module):
    def __init__(self, hidden_size, dropout=0.1) -> None:
        super(SublayerConnecttion, self).__init__()
        self.layer_norm = LayerNorm(x_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))


def clone_module_to_modulelist(module, module_num):
    return nn.ModuleList([deepcopy(module) for _ in range(module_num)])


class Crossmodal_Attention(nn.Module):
    def __init__(self, num_attention_head, hidden_size, intermediate_size, dropout_prob, layer_norm_eps) -> None:
        super(Crossmodal_Attention, self).__init__()
        self.attn = Attention(
            num_attention_head=num_attention_head,
            hidden_size=hidden_size,
            dropout_prob=dropout_prob
        )
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        self.sublayer_connection_list = clone_module_to_modulelist(
            module=SublayerConnecttion(
                hidden_size=hidden_size,
                dropout=dropout_prob
            ),
            module_num=2
        )

    def forward(self, query, key, value, encoder_attention_mask=None):
        x = self.sublayer_connection_list[0](
            query, lambda query: self.attn(
                query=query,
                key=key,
                value=value,
                encoder_attention_mask=encoder_attention_mask)[0]
        )
        return self.sublayer_connection_list[1](x, self.feed_forward)


class ModelPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.activation(self.dense(first_token_tensor))
        return pooled_output


class MSD_Noknowledge_Model(nn.Module):
    def __init__(self, args):
        super(MSD_Noknowledge_Model, self).__init__()
        self.args = args
        self.text_encoder = BertModel.from_pretrained(args.bert_name)
        self.vision_encoder = ViTModel.from_pretrained(args.vit_name)

        self.vision_proj = nn.Linear(
            in_features=self.vision_encoder.config.hidden_size,
            out_features=args.embed_dim
        )
        self.text_proj = nn.Linear(
            in_features=self.text_encoder.config.hidden_size,
            out_features=args.embed_dim
        )

        self.temp = nn.Parameter(torch.ones([]) * args.temp)     # temp
        self.queue_size = args.queue_size
        self.momentum = args.momentum
        self.loss_itc = 0.0

        # create momentum model
        self.text_encoder_m = BertModel.from_pretrained(args.bert_name)
        self.text_proj_m = nn.Linear(
            in_features=self.text_encoder_m.config.hidden_size,
            out_features=args.embed_dim
        )
        self.vision_encoder_m = ViTModel.from_pretrained(args.vit_name)
        self.vision_proj_m = nn.Linear(
            in_features=self.vision_encoder.config.hidden_size,
            out_features=args.embed_dim
        )

        self.model_pairs = [
            [self.vision_encoder, self.vision_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m]
        ]

        self.copy_params()

        # create the queue
        self.register_buffer('vision_queue', torch.randn(args.embed_dim, self.queue_size))
        self.register_buffer('text_queue', torch.randn(args.embed_dim, self.queue_size))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.vision_queue = F.normalize(input=self.vision_queue, dim=0)
        self.text_queue = F.normalize(input=self.text_queue, dim=0)

        # cross modal shared dictionary
        self.dict_feature = nn.Parameter(torch.randn(1, args.shared_space_dim, 768))

        self.dict_atten = Crossmodal_Attention(
            num_attention_head=self.text_encoder.config.num_attention_heads,
            hidden_size=self.text_encoder.config.hidden_size,
            intermediate_size=self.text_encoder.config.intermediate_size,
            dropout_prob=0.0,
            layer_norm_eps=self.text_encoder.config.layer_norm_eps
        )

        self.text_dict_cls = ModelPooler(self.text_encoder.config.hidden_size)
        self.vision_dict_cls = ModelPooler(self.text_encoder.config.hidden_size)

        # cross modal attention
        self.text2vit_atten = Crossmodal_Attention(
            num_attention_head=self.vision_encoder.config.num_attention_heads,
            hidden_size=self.vision_encoder.config.hidden_size,
            intermediate_size=self.vision_encoder.config.intermediate_size,
            dropout_prob=self.vision_encoder.config.hidden_dropout_prob,
            layer_norm_eps=self.vision_encoder.config.layer_norm_eps)

        self.vit2text_atten = Crossmodal_Attention(
            num_attention_head=self.text_encoder.config.num_attention_heads,
            hidden_size=self.text_encoder.config.hidden_size,
            intermediate_size=self.text_encoder.config.intermediate_size,
            dropout_prob=self.text_encoder.config.hidden_dropout_prob,
            layer_norm_eps=self.text_encoder.config.layer_norm_eps)

        # last linear classifier
        self.dropout = nn.Dropout(p=args.dropout_prob)
        self.first_classifier = nn.Linear(
            in_features=self.text_encoder.config.hidden_size * 2,
            out_features=self.text_encoder.config.hidden_size * 2 // 16
        )
        #self.activation = nn.Tanh()
        self.second_classifier = nn.Linear(
            in_features=self.text_encoder.config.hidden_size * 2 // 16,
            out_features=2
        )

    def forward(self, input_ids, token_type_ids, attention_mask, images, alpha=0, mode='train'):
        # text encoder, output are last_hidden_state and pooler_output
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        text_hidden_state, text_pooler_output = text_output.last_hidden_state, text_output.pooler_output
        text_feat = F.normalize(self.text_proj(text_pooler_output), dim=-1)

        # vision encoder, output are last_hidden_state and pooler_output
        vision_output = self.vision_encoder(images)
        vision_hidden_state, vision_pooler_output = vision_output.last_hidden_state, vision_output.pooler_output
        vision_feat = F.normalize(self.vision_proj(vision_pooler_output), dim=-1)

        # if mode is train, use momentum model for Global Contrastive Learning
        if mode == 'train':
            # get momentum features
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)  # 设置temp值区间范围min, max
                self._momentum_update()
                text_output_m = self.text_encoder_m(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True
                )
                text_hidden_state_m, text_pooler_output_m = text_output_m.last_hidden_state, \
                    text_output_m.pooler_output
                text_feat_m = F.normalize(self.text_proj_m(text_pooler_output_m), dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                vision_output_m = self.vision_encoder_m(images)
                vision_hidden_state_m, vision_pooler_output_m = vision_output_m.last_hidden_state, \
                    vision_output_m.pooler_output
                vision_feat_m = F.normalize(self.vision_proj_m(vision_pooler_output_m), dim=-1)
                vision_feat_all = torch.cat([vision_feat_m.t(), self.vision_queue.clone().detach()], dim=1)

                sim_i2t_m = vision_feat_m @ vision_feat_all / self.temp
                sim_t2i_m = text_feat_m @ text_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(self.args.device)
                sim_targets.fill_diagonal_(1)

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = vision_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ vision_feat_all / self.temp

            # compute KL loss
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            self.loss_itc = (loss_i2t + loss_t2i) / 2

            self._dequeue_and_enqueue(vision_feat_m, text_feat_m)


        # reconstruction text feature loss
        B, L, _ = text_hidden_state.shape
        text_dict_feature = self.dict_feature.repeat(B, 1, 1)
        text_memory_output = self.dict_atten(
            query=text_hidden_state,
            key=text_dict_feature,
            value=text_dict_feature
        )
        text_memory_cls = self.text_dict_cls(text_memory_output)
        text_similarity = torch.matmul(text_pooler_output, text_memory_cls.t())  # [B, B]
        text_targets = torch.zeros((B, B)).to(self.args.device)
        text_targets.fill_diagonal_(1)

        # reconstruction vision feature loss
        B, P, _ = vision_hidden_state.shape
        vision_dict_feature = self.dict_feature.repeat(B, 1, 1)
        vision_memory_output = self.dict_atten(
            query=vision_hidden_state,
            key=vision_dict_feature,
            value=vision_dict_feature
        )
        vision_memory_cls = self.vision_dict_cls(vision_memory_output)
        vision_similarity = torch.matmul(vision_pooler_output, vision_memory_cls.t())
        vision_targets = torch.zeros((B, B)).to(self.args.device)
        vision_targets.fill_diagonal_(1)

        # compute contrastive loss
        loss_t2t = -torch.sum(F.log_softmax(text_similarity, dim=1) * text_targets, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(vision_similarity, dim=1) * vision_targets, dim=1).mean()
        self.loss_md = (loss_t2t + loss_i2i) / 2

        # Cross-Modal Multi-head Attention
        vit2text_atten_output = self.vit2text_atten(
            query=text_memory_cls.unsqueeze(1),
            key=vision_memory_output,
            value=vision_memory_output
        )
        text2vit_atten_output = self.text2vit_atten(
            query=vision_memory_cls.unsqueeze(1),
            key=text_memory_output,
            value=text_memory_output,
            encoder_attention_mask=attention_mask.unsqueeze(1).unsqueeze(1)
        )

        # full-linear classifier
        output = torch.cat([vit2text_atten_output.squeeze(1), text2vit_atten_output.squeeze(1)], dim=-1)
        # output = self.second_classifier(self.activation(self.first_classifier(self.dropout(output))))
        output = self.second_classifier(self.dropout(self.first_classifier(output)))
        if mode == 'train':
            return output, self.loss_itc, self.loss_md
        return output, self.loss_md

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats):
        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.vision_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


class MSD_Knowledge_Model(nn.Module):
    def __init__(self, args):
        super(MSD_Knowledge_Model, self).__init__()
        self.args = args

        self.multimodal_clip_encoder = CLIPModel.from_pretrained(args.clip_name)
        self.vision_linear = nn.Linear(
            in_features=self.multimodal_clip_encoder.config.vision_config.hidden_size,
            out_features=args.embed_dim
        )
        self.text_linear = nn.Linear(
            in_features=self.multimodal_clip_encoder.config.text_config.hidden_size,
            out_features=args.embed_dim
        )

        self.temp = nn.Parameter(torch.ones([]) * args.temp)  # temp
        self.queue_size = args.queue_size
        self.momentum = args.momentum
        self.loss_itc, self.loss_md = 0.0, 0.0

        # create momentum model
        self.multimodal_clip_encoder_m = CLIPModel.from_pretrained(args.clip_name)
        self.text_linear_m = nn.Linear(
            in_features=self.multimodal_clip_encoder_m.config.text_config.hidden_size,
            out_features=args.embed_dim
        )
        self.vision_linear_m = nn.Linear(
            in_features=self.multimodal_clip_encoder_m.config.vision_config.hidden_size,
            out_features=args.embed_dim
        )

        self.model_pairs = [
            [self.multimodal_clip_encoder, self.multimodal_clip_encoder_m],
            [self.vision_linear, self.vision_linear_m],
            [self.text_linear, self.text_linear_m]
        ]

        self.copy_params()

        # create the queue
        self.register_buffer('vision_queue', torch.randn(args.embed_dim, self.queue_size))
        self.register_buffer('text_queue', torch.randn(args.embed_dim, self.queue_size))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.vision_queue = F.normalize(input=self.vision_queue, dim=0)
        self.text_queue = F.normalize(input=self.text_queue, dim=0)

        self.vit2vit_linear = nn.Linear(
            in_features=self.multimodal_clip_encoder.config.vision_config.hidden_size,
            out_features=self.multimodal_clip_encoder.config.projection_dim
        )

        # cross modal shared dictionary
        self.dict_feature = nn.Parameter(torch.randn(1, args.shared_space_dim, 768))

        self.dict_atten = Crossmodal_Attention(
            num_attention_head=self.multimodal_clip_encoder.config.text_config.num_attention_heads,
            hidden_size=self.multimodal_clip_encoder.config.text_config.hidden_size,
            intermediate_size=self.multimodal_clip_encoder.config.text_config.intermediate_size,
            dropout_prob=0.0,
            layer_norm_eps=self.multimodal_clip_encoder.config.text_config.layer_norm_eps
        )

        self.text_dict_cls = ModelPooler(self.multimodal_clip_encoder.config.text_config.hidden_size)
        self.vision_dict_cls = ModelPooler(self.multimodal_clip_encoder.config.text_config.hidden_size)

        # cross attention
        self.text2vit_atten = Crossmodal_Attention(
            num_attention_head=self.multimodal_clip_encoder.config.vision_config.num_attention_heads,
            hidden_size=self.multimodal_clip_encoder.config.projection_dim,
            intermediate_size=self.multimodal_clip_encoder.config.vision_config.intermediate_size,
            dropout_prob=self.multimodal_clip_encoder.config.vision_config.attention_dropout,
            layer_norm_eps=self.multimodal_clip_encoder.config.vision_config.layer_norm_eps
        )

        self.vit2text_atten = Crossmodal_Attention(
            num_attention_head=self.multimodal_clip_encoder.config.text_config.num_attention_heads,
            hidden_size=self.multimodal_clip_encoder.config.text_config.hidden_size,
            intermediate_size=self.multimodal_clip_encoder.config.text_config.intermediate_size,
            dropout_prob=self.multimodal_clip_encoder.config.text_config.attention_dropout,
            layer_norm_eps=self.multimodal_clip_encoder.config.text_config.layer_norm_eps
        )

        # last linear classifier
        self.dropout = nn.Dropout(p=args.dropout_prob)
        self.first_classifier = nn.Linear(
            in_features=self.multimodal_clip_encoder.config.projection_dim * 2,
            out_features=self.multimodal_clip_encoder.config.projection_dim * 2 // 16
        )
        # self.activation = nn.Tanh()
        self.second_classifier = nn.Linear(
            in_features=self.multimodal_clip_encoder.config.projection_dim * 2 // 16,
            out_features=2
        )

    def forward(self, input_ids, attention_mask, images, alpha=0, mode='train'):
        clip_output = self.multimodal_clip_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
            return_dict=True
        )
        text_pooler_output, vision_pooler_output = clip_output.text_embeds, clip_output.image_embeds

        text_hidden_state = clip_output.text_model_output.last_hidden_state
        vision_hidden_state = clip_output.vision_model_output.last_hidden_state

        text_feat = F.normalize(self.text_linear(text_pooler_output), dim=-1)
        vision_feat = F.normalize(self.vision_linear(vision_hidden_state[:, 0, :]), dim=-1)

        # if mode is train, use momentum model for Global Contrastive Learning
        if mode == 'train':
            # get momentum features
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)  # 设置temp值区间范围min, max
                self._momentum_update()
                clip_output_m = self.multimodal_clip_encoder_m(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=images,
                    return_dict=True
                )
                text_pooler_output_m, vision_pooler_output_m = clip_output_m.text_embeds, clip_output_m.image_embeds

                text_hidden_state_m = clip_output_m.text_model_output.last_hidden_state
                vision_hidden_state_m = clip_output_m.vision_model_output.last_hidden_state

                text_feat_m = F.normalize(self.text_linear_m(text_pooler_output_m), dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                vision_feat_m = F.normalize(self.vision_linear_m(vision_hidden_state_m[:, 0, :]), dim=-1)
                vision_feat_all = torch.cat([vision_feat_m.t(), self.vision_queue.clone().detach()], dim=1)

                sim_i2t_m = vision_feat_m @ vision_feat_all / self.temp
                sim_t2i_m = text_feat_m @ text_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(self.args.device)
                sim_targets.fill_diagonal_(1)

                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = vision_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ vision_feat_all / self.temp

            # compute KL loss
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            self.loss_itc = (loss_i2t + loss_t2i) / 2

            self._dequeue_and_enqueue(vision_feat_m, text_feat_m)

        vision_hidden_state = self.vit2vit_linear(vision_hidden_state)  # dim 1024 -> 768
        # text_hidden_state = text_hidden_state[:, 1:]
        # vision_hidden_state = vision_hidden_state[:, 1:]
        # attention_mask = attention_mask[:, 1:]

        # reconstruction text feature loss
        B, L, _ = text_hidden_state.shape
        text_dict_feature = self.dict_feature.repeat(B, 1, 1)
        text_memory_output = self.dict_atten(
            query=text_hidden_state,
            key=text_dict_feature,
            value=text_dict_feature
        )
        text_memory_cls = self.text_dict_cls(text_memory_output)
        text_similarity = torch.matmul(text_pooler_output, text_memory_cls.t())  # [B, B]
        text_targets = torch.zeros((B, B)).to(self.args.device)
        text_targets.fill_diagonal_(1)

        # reconstruction vision feature loss
        B, P, _ = vision_hidden_state.shape
        vision_dict_feature = self.dict_feature.repeat(B, 1, 1)
        vision_memory_output = self.dict_atten(
            query=vision_hidden_state,
            key=vision_dict_feature,
            value=vision_dict_feature
        )
        vision_memory_cls = self.vision_dict_cls(vision_memory_output)
        vision_similarity = torch.matmul(vision_pooler_output, vision_memory_cls.t())
        vision_targets = torch.zeros((B, B)).to(self.args.device)
        vision_targets.fill_diagonal_(1)

        # compute contrastive loss
        loss_t2t = -torch.sum(F.log_softmax(text_similarity, dim=1) * text_targets, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(vision_similarity, dim=1) * vision_targets, dim=1).mean()
        self.loss_md = (loss_t2t + loss_i2i) / 2

        # Cross-Modal Multi-head Attention
        vit2text_atten_output = self.vit2text_atten(
            query=text_memory_cls.unsqueeze(1),
            key=vision_memory_output,
            value=vision_memory_output
        )
        text2vit_atten_output = self.text2vit_atten(
            query=vision_memory_cls.unsqueeze(1),
            key=text_memory_output,
            value=text_memory_output,
            encoder_attention_mask=attention_mask.unsqueeze(1).unsqueeze(1)
        )

        # full-linear classifier
        output = torch.cat([vit2text_atten_output.squeeze(1), text2vit_atten_output.squeeze(1)], dim=-1)
        output = self.second_classifier(self.dropout(self.first_classifier(output)))
        # output = self.second_classifier(self.activation(self.first_classifier(output)))
        if mode == 'train':
            return output, self.loss_itc, self.loss_md
        return output, self.loss_md

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats):
        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.vision_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device
                     ).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(
        b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device
                       ) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2)/beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad,
                           beta=0.5, iteration=50, k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
             beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance

