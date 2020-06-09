
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

import numpy as np

from transformers.modeling_bert import BertPooler, BertModel, BertSelfAttention, BertForSequenceClassification, BertLayerNorm


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class BertPoolerBase(BertPooler):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # this we don't have in default BertPooler
        self.distribution = config.distribution
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.pooler_activation == 'tanh':
            self.pooler_activation = nn.Tanh()
        elif config.pooler_activation == 'relu':
            self.pooler_activation = nn.ReLU()
        elif config.pooler_activation == 'gelu':
            self.pooler_activation = F.gelu
        else:
            raise KeyError(f'Unknown activation: {config.pooler_activation}')

    def reset_parameters(self):
        print(f'Re-initializing pooler weights from {self.distribution} distribution')

        if self.distribution == 'uniform':
            print(f'bound: {self.config.distribution_bound}')
            self.dense.weight.data.uniform_(-self.config.distribution_bound, self.config.distribution_bound)
            self.dense.bias.data.uniform_(-self.config.distribution_bound, self.config.distribution_bound)
            # self.dense.bias.data.zero_()
        elif self.distribution == 'normal':
            print(f'std: {self.config.distribution_std}')
            # BERT initializes linear layers with: module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # where self.config.initializer_range = 0.02
            self.dense.weight.data.normal_(mean=0.0, std=self.config.distribution_std)
            self.dense.bias.data.zero_()
        else:
            raise KeyError(f"Unknown distribution {self.distribution}")


class BertCLSPooler(BertPoolerBase):
    def __init__(self, config):
        super().__init__(config)
        self.batch_id = 0

    def forward(self, hidden_states, attention_mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token ([CLS])
        token_tensor = hidden_states[:, 0]

        # Save token_tensor to disk
        # token_tensor_np = token_tensor.detach().cpu().numpy()
        # filename = '/logfiles/dodge-et-al-2020/' + f'token_tensor_{self.batch_id}.npy'
        # np.save(filename, token_tensor_np, allow_pickle=True)
        # self.batch_id += 1

        # RoBERTa uses an additional dropout here (before the linear transformation)
        if self.config.pooler_dropout:
            token_tensor_dropout = self.dropout(token_tensor)  # this we don't have in default BertPooler
        else:
            token_tensor_dropout = token_tensor

        pooled_linear_transform = self.dense(token_tensor_dropout)

        if self.config.pooler_layer_norm:  # apply LayerNorm to tanh pre-activations
            normalized_pooled_linear_transform = self.LayerNorm(pooled_linear_transform)
        else:
            normalized_pooled_linear_transform = pooled_linear_transform

        pooled_activation = self.pooler_activation(normalized_pooled_linear_transform)

        return pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor


class BertModelWithPooler(BertModel):
    def __init__(self, config):
        super().__init__(config)

        # Choose pooler based on config
        if config.pooler == 'cls':
            self.pooler = BertCLSPooler(config)
        else:
            raise KeyError(f'Unknown pooler: {config.pooler}')

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
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

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
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
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]

        pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor = self.pooler(sequence_output, attention_mask)
        # pooled activation is the results of the applying the pooler's tanh
        # pooler_output is the input to the pooler's tanh
        # token_tensor is the pooled vector, either CLS, 5th token, or mean over tokens

        outputs = (sequence_output, pooled_activation, normalized_pooled_linear_transform, pooled_linear_transform, token_tensor) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_activation, pooled_linear_transform, token_tensor, (hidden_states), (attentions)


class BertForSequenceClassificationWithPooler(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelWithPooler(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

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
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

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

        return outputs

    def manually_init_weights(self, std):
        # Initialize weights following: https://arxiv.org/abs/2002.06305
        print(f'Initializing weights of linear classifier: mean = 0.0, std = {std}')
        self.classifier.weight.data.normal_(mean=0.0, std=std)
        self.classifier.bias.data.zero_()
        # self.classifier.bias.data.normal_(mean=0.0, std=std)
