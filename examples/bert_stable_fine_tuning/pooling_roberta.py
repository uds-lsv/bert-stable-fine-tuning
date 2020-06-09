
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_roberta import RobertaEmbeddings, RobertaModel, RobertaClassificationHead, RobertaForSequenceClassification
from transformers.modeling_bert import BertLayerNorm


def cls_pooler(features):
    return features[:, 0, :]  # take <s> token (equiv. to [CLS])


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class RobertaClassificationHeadWithPooler(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.distribution = config.distribution
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.pooler == 'cls':
            self.pooler = cls_pooler
        else:
            raise KeyError(f'Unknown pooler: {config.pooler}')

        if config.pooler_activation == 'tanh':
            self.pooler_activation = nn.Tanh()
        elif config.pooler_activation == 'relu':
            self.pooler_activation = nn.ReLU()
        elif config.pooler_activation == 'gelu':
            self.pooler_activation = F.gelu
        else:
            raise KeyError(f'Unknown activation: {config.pooler_activation}')

    def forward(self, features, **kwargs):
        # pooling
        token_tensor = self.pooler(features)  # apply pooler

        if self.config.pooler_dropout:
            token_tensor_dropout = self.dropout(token_tensor)  # this we don't have in BERT pooler
        else:
            token_tensor_dropout = token_tensor

        pooled_linear_transform = self.dense(token_tensor_dropout)

        if self.config.pooler_layer_norm:  # apply LayerNorm to tanh pre-activations
            normalized_pooled_linear_transform = self.layer_norm(pooled_linear_transform)
        else:
            normalized_pooled_linear_transform = pooled_linear_transform

        pooled_activation = self.pooler_activation(normalized_pooled_linear_transform)

        # classifier
        pooled_activation_dropout = self.dropout(pooled_activation)
        x = self.out_proj(pooled_activation_dropout)

        return x, pooled_activation_dropout, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor

    def reset_pooler_parameters(self):
        print(f'Re-initializing pooler weights from {self.distribution} distribution')

        if self.distribution == 'uniform':
            bound = 1 / math.sqrt(self.config.distribution_fan_in)
            self.dense.weight.data.uniform_(-bound, bound)
            self.dense.bias.data.uniform_(-bound, bound)
            # self.dense.bias.data.zero_()
        elif self.distribution == 'normal':
            # BERT initializes linear layers with: module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # where self.config.initializer_range = 0.02
            self.dense.weight.data.normal_(mean=0.0, std=self.config.distribution_std)
            self.dense.bias.data.zero_()
        else:
            raise KeyError(f"Unknown distribution {self.distribution}")


class RobertaForSequenceClassificationWithPooler(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = RobertaClassificationHeadWithPooler(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits, pooled_activation_dropout, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor = self.classifier(sequence_output)

        outputs = (logits, pooled_activation_dropout, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, pooled_activation, pooled_linear_transform, token_tensor, (hidden_states), (attentions)

    def manually_init_weights(self, std):
        # Initialize weights following: https://arxiv.org/abs/2002.06305
        print(f'Initializing weights of linear classifier: mean = 0.0, std = {std}')
        self.classifier.out_proj.weight.data.normal_(mean=0.0, std=std)
        self.classifier.out_proj.bias.data.zero_()
