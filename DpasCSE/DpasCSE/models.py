import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


###############################################################################
# class PrefixEncoder(torch.nn.Module):
#     r'''
#     The torch.nn model to encode the prefix
#     Input shape: (batch-size, prefix-length)
#     Output shape: (batch-size, prefix-length, 2*layers*hidden)
#     '''
#     def __init__(self, config, model_args):
#         super().__init__()
#         self.prefix_projection = model_args.prefix_projection
#         if self.prefix_projection:
#             # Use a two-layer MLP to encode the prefix
#             self.embedding = torch.nn.Embedding(model_args.pre_seq_len, config.hidden_size)
#             self.trans = torch.nn.Sequential(
#                 torch.nn.Linear(config.hidden_size, model_args.prefix_hidden_size),
#                 torch.nn.Tanh(),
#                 torch.nn.Linear(model_args.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
#             )
#         else:
#             self.embedding = torch.nn.Embedding(model_args.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)
#
#     def forward(self, prefix: torch.Tensor):
#         if self.prefix_projection:
#             prefix_tokens = self.embedding(prefix)
#             past_key_values = self.trans(prefix_tokens)
#         else:
#             past_key_values = self.embedding(prefix)
#         return past_key_values
###############################################################################
########################################################
#尝试利用让前缀编码器使用自注意力机制捕捉长距离依赖关系
class PrefixEncoder(torch.nn.Module):
    def __init__(self, config, model_args):
        super().__init__()
        self.config = config  # 将config保存为实例属性
        self.prefix_projection = model_args.prefix_projection
        self.max_num_heads = 16  # 设置最大头数
        self.current_num_heads = 4  # 初始头数为4
        self.head_growing_patience = 2  # 当验证损失没有改善时,等待的epoch数
        self.wait = 0  # 计数器,用于追踪验证损失没有改善的epoch数
        self.best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大

        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(model_args.pre_seq_len, self.config.hidden_size)
            self.attention = torch.nn.MultiheadAttention(self.config.hidden_size, num_heads=self.current_num_heads, batch_first=True)
            self.trans = torch.nn.Linear(self.config.hidden_size, self.config.num_hidden_layers * 2 * self.config.hidden_size)
        else:
            self.embedding = torch.nn.Embedding(model_args.pre_seq_len, self.config.num_hidden_layers * 2 * self.config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            prefix_tokens, _ = self.attention(prefix_tokens, prefix_tokens, prefix_tokens)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

    def update_num_heads(self, val_loss):
        """
        根据验证损失,更新头数量
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.head_growing_patience and self.current_num_heads < self.max_num_heads:
                self.current_num_heads += 4  # 每次增加4个头
                self.attention = torch.nn.MultiheadAttention(self.config.hidden_size, num_heads=self.current_num_heads, batch_first=True)
                print(f'Increased number of heads to {self.current_num_heads}')
                self.wait = 0


#########################################################



class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    目的是将输入的特征通过一个全连接层和一个激活函数进行变换，得到新的特征表示。这种结构通常被称为多层感知器（MLP），是神经网络中常见的一种结构。
    在这个模型中，激活函数选择的是Tanh函数，它可以将输入的数值压缩到-1到1之间，有助于控制输出的数值范围，防止梯度爆炸或消失
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    计算余弦相似度
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    ###########################################################################
    if cls.model_args.do_eh_loss:
        cls.margin_rank_loss = nn.MarginRankingLoss(margin=cls.model_args.eh_loss_margin)
    ###########################################################################
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    past_key_values=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    ##########################################################################
    past_key_values = cls.get_prompt(batch_size=input_ids.shape[0])
        #past_key_values是transformer模型的隐藏状态，它在transformer模型的多层中传递，用于捕获上下文信息。
    prefix_attention_mask = torch.ones(input_ids.shape[0], cls.pre_seq_len).to(cls.device)
        #prefix_attention_mask是transformer模型的注意力掩码，它用于控制哪些位置的token可以被注意到。
    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
       #attention_mask是transformer模型的注意力权重，它用于计算每个位置的token与其他位置的token之间的注意力程度。
    ##########################################################################

    # Get raw embeddings
    # outputs = encoder(
    #     input_ids,
    #     attention_mask=attention_mask,
    #     token_type_ids=token_type_ids,
    #     position_ids=position_ids,
    #     head_mask=head_mask,
    #     inputs_embeds=inputs_embeds,
    #     output_attentions=output_attentions,
    #     output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
    #     return_dict=True,
    #     past_key_values=past_key_values, # new added
    #
    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        past_key_values=past_key_values,
    )

    # 从outputs中提取最后一层隐藏状态的张量
    last_hidden_state = outputs.last_hidden_state

    # 添加噪声
    noise = torch.randn_like(last_hidden_state) * 0.2
    last_hidden_state = last_hidden_state + noise

    # 重新包装outputs
    outputs = BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=last_hidden_state,
        pooler_output=outputs.pooler_output,
        hidden_states=outputs.hidden_states
    )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)


    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
            past_key_values=past_key_values, # new added
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs) # ÕâÀïattention maskÐèÒªÐÞ¸Ä
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        #######################################################################
        if cls.model_args.do_eh_loss:
            z1_z2_cos = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            anchor_pos = torch.diag(z1_z2_cos).unsqueeze(dim=-1)

            vector = -1e9 * torch.ones(z1_z2_cos.size(0)).to(cls.device)
            mask = torch.diag(torch.ones_like(vector)).to(cls.device)
            z1_z2_cos_new = mask * torch.diag(vector) + (1. - mask) * z1_z2_cos
            max_s3_sn = torch.max(torch.cat([z1_z2_cos_new, z1_z3_cos], dim=-1), -1, keepdim=True)[0]

            rank_label = torch.ones_like(anchor_pos).long()
            margin_loss = cls.margin_rank_loss(anchor_pos * cls.model_args.temp,
                                               max_s3_sn * cls.model_args.temp,
                                               rank_label)
        #######################################################################


    # labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    # loss_fct = nn.CrossEntropyLoss()
    #
    # # Calculate loss with hard negatives
    # if num_sent == 3:
    #     # Note that weights are actually logits of weights
    #     z3_weight = cls.model_args.hard_negative_weight
    #     weights = torch.tensor(
    #         [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    #     ).to(cls.device)
    #     cos_sim = cos_sim + weights
    #
    # loss = loss_fct(cos_sim, labels)
    # 计算相似度矩阵
    sim_matrix = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    # 构造标签
    labels = torch.arange(sim_matrix.size(0)).long().to(cls.device)

    # 计算对比损失
    loss = F.cross_entropy(sim_matrix, labels)

    # 如果有hard negative,将其也考虑在内
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))

        # 引入可学习的z3_weight参数
        z3_weight = torch.tensor(cls.model_args.hard_negative_weight, requires_grad=True).to(cls.device)
        weighted_z1_z3_cos = z1_z3_cos * z3_weight

        sim_matrix = torch.cat([sim_matrix, weighted_z1_z3_cos], 1)
        loss = F.cross_entropy(sim_matrix, labels)
    ###########################################################################
    if cls.model_args.do_eh_loss:
        loss += cls.model_args.eh_loss_weight * margin_loss
    ###########################################################################

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = (loss, cls.model_args.mlm_weight * masked_lm_loss)

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    ##########################################################################
    past_key_values = cls.get_prompt(batch_size=input_ids.shape[0])
    prefix_attention_mask = torch.ones(input_ids.shape[0], cls.pre_seq_len).to(cls.device)
    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
    ##########################################################################

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
        past_key_values=past_key_values, # new added
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        ######################################################################
        self.pre_seq_len = self.model_args.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config, self.model_args)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.prefix_encoder.parameters():
            param.requires_grad = True

        # compute the number of total parameters and tunable parameters
        total_param = sum(p.numel() for p in self.parameters())
        trainable_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('total param is {}, trainable param is {}'.format(total_param, trainable_param))
        ######################################################################

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    ##########################################################################
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    ##########################################################################

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        ######################################################################
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = self.model_args.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config, self.model_args)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # compute the number of total parameters and tunable parameters
        total_param = sum(p.numel() for p in self.parameters())
        trainable_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('total param is {}, trainable param is {}'.format(total_param, trainable_param))
        ######################################################################

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    ##########################################################################
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    ##########################################################################

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
