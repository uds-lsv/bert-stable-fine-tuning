
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_albert import AlbertModel, AlbertForSequenceClassification
from transformers.modeling_bert import BertLayerNorm


def cls_pooler(features):
    return features[:, 0]  # take <s> token (equiv. to [CLS])


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class AlbertModelWithPooler(AlbertModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.pooler == 'cls':
            self.pooler_fct = cls_pooler
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        # pooling
        token_tensor = self.pooler_fct(sequence_output)  # apply pooler

        if self.config.pooler_dropout:
            token_tensor_dropout = self.dropout(token_tensor)  # this we don't have in BERT pooler
        else:
            token_tensor_dropout = token_tensor

        pooled_linear_transform = self.pooler(token_tensor_dropout)  # for ALBERT the name of the pooler transform is pooler

        if self.config.pooler_layer_norm:  # apply LayerNorm to tanh pre-activations
            normalized_pooled_linear_transform = self.layer_norm(pooled_linear_transform)
        else:
            normalized_pooled_linear_transform = pooled_linear_transform

        pooled_activation = self.pooler_activation(normalized_pooled_linear_transform)

        outputs = (sequence_output, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        return outputs

    def reset_pooler_parameters(self):
        print(f'Re-initializing pooler weights from {self.config.distribution} distribution')

        if self.config.distribution == 'uniform':
            bound = 1 / math.sqrt(self.config.distribution_bound)
            self.pooler.weight.data.uniform_(-bound, bound)
            self.pooler.bias.data.uniform_(-bound, bound)
            # self.pooler.bias.data.zero_()
        elif self.config.distribution == 'normal':
            # BERT initializes linear layers with: module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # where self.config.initializer_range = 0.02
            self.pooler.weight.data.normal_(mean=0.0, std=self.config.distribution_std)
            self.pooler.bias.data.zero_()
        else:
            raise KeyError(f"Unknown distribution {self.config.distribution}")


class AlbertForSequenceClassificationWithPooler(AlbertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModelWithPooler(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

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
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor = outputs[1:5]  # get outputs from the pooler
        pooled_activation_dropout = self.dropout(pooled_activation)
        logits = self.classifier(pooled_activation_dropout)

        outputs = (logits, pooled_activation_dropout, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor) + outputs[5:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def manually_init_weights(self, std):
        # Initialize weights following: https://arxiv.org/abs/2002.06305
        print(f'Initializing weights of linear classifier: mean = 0.0, std = {std}')
        self.classifier.weight.data.normal_(mean=0.0, std=std)
        self.classifier.bias.data.zero_()
        # self.classifier.bias.data.normal_(mean=0.0, std=std)
