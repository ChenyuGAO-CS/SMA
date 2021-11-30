# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Update kwargs with defaults
import os
import pickle
from functools import lru_cache
import functools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pythia.modules.attention import AttentionLayer
from pythia.modules.layers import Identity
from pythia.utils.vocab import Vocab
from pythia.modules.layers import GatedTanh, ModalCombineLayer, TransformLayer


class TextEmbedding(nn.Module):
    def __init__(self, emb_type, **kwargs):
        super(TextEmbedding, self).__init__()
        self.model_data_dir = kwargs.get("model_data_dir", None)
        self.embedding_dim = kwargs.get("embedding_dim", None)

        # Update kwargs here
        if emb_type == "identity":
            self.module = Identity()
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "vocab":
            self.module = VocabEmbedding(**kwargs)
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "preextracted":
            self.module = PreExtractedEmbedding(**kwargs)
        elif emb_type == "bilstm":
            self.module = BiLSTMTextEmbedding(**kwargs)
        elif emb_type == "attention":
            self.module = AttentionTextEmbedding(**kwargs)
        elif emb_type == "torch":
            vocab_size = kwargs["vocab_size"]
            embedding_dim = kwargs["embedding_dim"]
            self.module = nn.Embedding(vocab_size, embedding_dim)
            self.module.text_out_dim = self.embedding_dim
        else:
            raise NotImplementedError("Unknown question embedding '%s'" % emb_type)

        self.text_out_dim = self.module.text_out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class VocabEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_params):
        self.vocab = Vocab(**vocab_params)
        self.module = self.vocab.get_embedding(nn.Embedding, embedding_dim)

    def forward(self, x):
        return self.module(x)


class BiLSTMTextEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        bidirectional=False,
        rnn_type="GRU",
    ):
        super(BiLSTMTextEmbedding, self).__init__()
        self.text_out_dim = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU

        self.recurrent_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        # Return last state
        if self.bidirectional:
            return out[:, -1]

        forward_ = out[:, -1, : self.num_hid]
        backward = out[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output


class PreExtractedEmbedding(nn.Module):
    def __init__(self, out_dim, base_path):
        super(PreExtractedEmbedding, self).__init__()
        self.text_out_dim = out_dim
        self.base_path = base_path
        self.cache = {}

    def forward(self, qids):
        embeddings = []
        for qid in qids:
            embeddings.append(self.get_item(qid))
        return torch.stack(embeddings, dim=0)

    @lru_cache(maxsize=5000)
    def get_item(self, qid):
        return np.load(os.path.join(self.base_path, str(qid.item()) + ".npy"))


class AttentionTextEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super(AttentionTextEmbedding, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU()

        self.conv1_oo = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2_oo = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu_oo = nn.ReLU()

        self.conv1_ot = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2_ot = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu_ot = nn.ReLU()

        self.conv1_text = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2_text = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu_text = nn.ReLU()

        self.conv1_tt = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2_tt = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu_tt = nn.ReLU()

        self.conv1_to = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2_to = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu_to = nn.ReLU()


    def forward(self, x):
        batch_size = x.size(0)

        bert_reshape = x.permute(0,2,1) # N * hidden_dim * T

        qatt_conv1 = self.conv1(bert_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        """obj-obj self-attention"""
        qatt_conv1_oo = self.conv1_oo(bert_reshape)  # N x conv1_out x T
        qatt_relu_oo = self.relu_oo(qatt_conv1_oo)
        qatt_conv2_oo = self.conv2_oo(qatt_relu_oo)  # N x conv2_out x T

        # Over last dim
        qtt_softmax_oo = nn.functional.softmax(qatt_conv2_oo, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature_oo = torch.bmm(qtt_softmax_oo, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat_oo = qtt_feature_oo.view(batch_size, -1)

        """obj-text self-attention"""
        qatt_conv1_ot = self.conv1_ot(bert_reshape)  # N x conv1_out x T
        qatt_relu_ot = self.relu_ot(qatt_conv1_ot)
        qatt_conv2_ot = self.conv2_ot(qatt_relu_ot)  # N x conv2_out x T

        # Over last dim
        qtt_softmax_ot = nn.functional.softmax(qatt_conv2_ot, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature_ot = torch.bmm(qtt_softmax_ot, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat_ot = qtt_feature_ot.view(batch_size, -1)

        """text self-attention"""
        qatt_conv1_text = self.conv1_text(bert_reshape)  # N x conv1_out x T
        qatt_relu_text = self.relu_text(qatt_conv1_text)
        qatt_conv2_text = self.conv2_text(qatt_relu_text)  # N x conv2_out x T

        # Over last dim
        qtt_softmax_text = nn.functional.softmax(qatt_conv2_text, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature_text = torch.bmm(qtt_softmax_text, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat_text = qtt_feature_text.view(batch_size, -1)

        """text-text self-attention"""
        qatt_conv1_tt = self.conv1_tt(bert_reshape)  # N x conv1_out x T
        qatt_relu_tt = self.relu_tt(qatt_conv1_tt)
        qatt_conv2_tt = self.conv2_tt(qatt_relu_tt)  # N x conv2_out x T

        # Over last dim
        qtt_softmax_tt = nn.functional.softmax(qatt_conv2_tt, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature_tt = torch.bmm(qtt_softmax_tt, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat_tt = qtt_feature_tt.view(batch_size, -1)

        """text-text self-attention"""
        qatt_conv1_to = self.conv1_to(bert_reshape)  # N x conv1_out x T
        qatt_relu_to = self.relu_to(qatt_conv1_to)
        qatt_conv2_to = self.conv2_to(qatt_relu_to)  # N x conv2_out x T

        # Over last dim
        qtt_softmax_to = nn.functional.softmax(qatt_conv2_to, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature_to = torch.bmm(qtt_softmax_to, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat_to = qtt_feature_to.view(batch_size, -1)

        return x, qtt_feature_concat, qtt_feature_concat_oo, qtt_feature_concat_ot, qtt_feature_concat_text, qtt_feature_concat_tt, qtt_feature_concat_to, qtt_softmax, qtt_softmax_oo, qtt_softmax_ot, qtt_softmax_text, qtt_softmax_tt, qtt_softmax_to



class ImageEmbedding(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    ques_embed:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    def __init__(self, img_dim, question_dim, **kwargs):
        super(ImageEmbedding, self).__init__()
        combine_type = kwargs["modal_combine"]["type"]
        combine_params = kwargs["modal_combine"]["params"]
        transform_type = kwargs["transform"]["type"]
        transform_params = kwargs["transform"]["params"]
        modal_combine_layer = ModalCombineLayer(
            combine_type, img_dim, question_dim, **combine_params
        )
        hetero_modal_combine_layer = ModalCombineLayer(
            combine_type, img_dim, question_dim, **combine_params
        )
        transform_layer = TransformLayer(
            transform_type, modal_combine_layer.out_dim, **transform_params
        )
        hetero_transform_layer = TransformLayer(
            transform_type, modal_combine_layer.out_dim, **transform_params
        )

        # prior weight w
        self.prior_weight_fc = nn.Linear(768, 2) # question's max length
        self.prior_weight_fc_3 = nn.Linear(768, 3)
        self.prior_weight_fc_3_text = nn.Linear(768, 3)
        self.vis_dim = 2048  # the dim of input image's visual feature: v
        self.jemb_dim = 2048  # the dim of all processed features are same
        self.jemb_drop_out = 0.4
        self.edge_loc_dim = 5
        self.text_dim = 300
        self.jemb_dim_text = 300

        self.modal_combine_layer = modal_combine_layer
        self.hetero_modal_combine_layer = hetero_modal_combine_layer
        self.transform_layer = transform_layer
        self.hetero_transform_layer = hetero_transform_layer

        self.image_attention_model = AttentionLayer(img_dim, question_dim, **kwargs)
        self.out_dim = self.image_attention_model.out_dim

        # MLP for embedding edge features(concatenation of 5d edge feature and neighbor node feature)
        # neighbor might be obj or OCR
        self.edge_oo_emb = nn.Sequential(nn.Linear(self.edge_loc_dim + self.vis_dim, self.jemb_dim),
                                        nn.BatchNorm1d(self.jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.jemb_drop_out),
                                        nn.Linear(self.jemb_dim, self.jemb_dim),
                                        nn.BatchNorm1d(self.jemb_dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(self.jemb_dim)
                                        )
        self.edge_ot_emb = nn.Sequential(nn.Linear(self.edge_loc_dim + self.text_dim, self.jemb_dim),
                                        nn.BatchNorm1d(self.jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.jemb_drop_out),
                                        nn.Linear(self.jemb_dim, self.jemb_dim),
                                        nn.BatchNorm1d(self.jemb_dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(self.jemb_dim)
                                        )
        self.edge_tt_emb = nn.Sequential(nn.Linear(self.edge_loc_dim + self.text_dim, self.jemb_dim_text),
                                              nn.BatchNorm1d(self.jemb_dim_text),
                                              nn.ReLU(),
                                              nn.Dropout(self.jemb_drop_out),
                                              nn.Linear(self.jemb_dim_text, self.jemb_dim_text),
                                              nn.BatchNorm1d(self.jemb_dim_text),
                                              nn.ReLU(),
                                              nn.BatchNorm1d(self.jemb_dim_text)
                                              )
        self.edge_to_emb = nn.Sequential(nn.Linear(self.edge_loc_dim + self.vis_dim, self.jemb_dim_text),
                                        nn.BatchNorm1d(self.jemb_dim_text),
                                        nn.ReLU(),
                                        nn.Dropout(self.jemb_drop_out),
                                        nn.Linear(self.jemb_dim_text, self.jemb_dim_text),
                                        nn.BatchNorm1d(self.jemb_dim_text),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(self.jemb_dim_text)
                                        )

    def forward(self, attr, image_feat_variable, ques_embed, image_dims, 
                s_homo=None, homo_edge_feature=None, 
                s_hetero=None, hetero_edge_feature=None,
                pre_ques_embed=None
                ):
        # N x K x n_att
        if attr == "image":
            s_oo, s_ot = s_homo, s_hetero
            oo_edge_feature, ot_edge_feature = homo_edge_feature, hetero_edge_feature
            if (s_oo is not None) and (oo_edge_feature is not None) and \
                (s_ot is not None) and (ot_edge_feature is not None) and \
                (pre_ques_embed is not None):
                batch_size, node_num, _ = image_feat_variable.shape  # torch.Size([32, 36, 2048])
            
                oo_feat_variable = self.GenEdgeFeature(False, 'obj', image_feat_variable, oo_edge_feature, s_oo)
                ot_feat_variable = self.GenEdgeFeature(True, 'obj', image_feat_variable, ot_edge_feature, s_ot)

                p_o, p_oo, p_ot = self.image_attention_model(
                    attr, image_feat_variable, ques_embed, image_dims,
                    oo_edge_feature=oo_feat_variable, s_oo=s_oo,
                    ot_edge_feature=ot_feat_variable, s_ot=s_ot,
                )  # torch.Size([128, 100, 1])
                attn_weight = F.softmax(self.prior_weight_fc_3(pre_ques_embed.mean(1)), dim=1)  # torch.Size([128, 3])
                attn_weight_o = attn_weight[:, 0].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_weight_oo = attn_weight[:, 1].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_weight_ot = attn_weight[:, 2].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_temp_o = attn_weight_o * p_o + attn_weight_oo * p_oo + attn_weight_ot * p_ot # torch.Size([128, 100, 1])

                x_obj = image_feat_variable
                g_o = torch.bmm(attn_temp_o.permute(0,2,1), x_obj)
                
                return g_o

            elif (s_oo is not None) and (oo_edge_feature is not None) and (pre_ques_embed is not None):
                batch_size, node_num, _ = image_feat_variable.shape  # torch.Size([32, 36, 2048])
                
                oo_feat_variable = self.GenEdgeFeature(False, 'obj', image_feat_variable, oo_edge_feature, s_oo)

                attention, attention_obj = self.image_attention_model(
                    attr, image_feat_variable, ques_embed, image_dims,
                    oo_edge_feature=oo_feat_variable,
                    s_oo=s_oo
                )  # torch.Size([128, 100, 1])
                attn_weight = F.softmax(self.prior_weight_fc(pre_ques_embed.mean(1)), dim=1)
                attn_weight_exp_1 = attn_weight[:, 0].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_weight_exp_2 = attn_weight[:, 1].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_temp = attn_weight_exp_1 * attention + attn_weight_exp_2 * attention_obj
                # att_reshape = attention.permute(0, 2, 1)  # torch.Size([128, 1, 100])
                att_reshape = attn_temp.permute(0, 2, 1)  # torch.Size([128, 1, 100])

                tmp_embedding = torch.bmm(
                    att_reshape, image_feat_variable
                )  # N x n_att x image_dim, torch.Size([128, 1, 2048])
                batch_size = att_reshape.size(0)
                image_embedding = tmp_embedding.view(batch_size, -1)

                return image_embedding, attn_temp

            elif (s_ot is not None) and (ot_edge_feature is not None) and (pre_ques_embed is not None):
                batch_size, node_num, _ = image_feat_variable.shape  # torch.Size([32, 36, 2048])
                
                ot_feat_variable = self.GenEdgeFeature(True, 'obj', image_feat_variable, ot_edge_feature,
                                                            s_ot)

                attention, attention_obj = self.image_attention_model(
                    attr, image_feat_variable, ques_embed, image_dims,
                    oo_edge_feature=ot_feat_variable,
                    s_oo=s_ot
                )  # torch.Size([128, 100, 1])
                attn_weight = F.softmax(self.prior_weight_fc(pre_ques_embed.mean(1)), dim=1)  # torch.Size([128, 2])
                attn_weight_exp_1 = attn_weight[:, 0].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_weight_exp_2 = attn_weight[:, 1].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_temp = attn_weight_exp_1 * attention + attn_weight_exp_2 * attention_obj
                # att_reshape = attention.permute(0, 2, 1)  # torch.Size([128, 1, 100])
                att_reshape = attn_temp.permute(0, 2, 1)  # torch.Size([128, 1, 100])

                tmp_embedding = torch.bmm(
                    att_reshape, image_feat_variable
                )  # N x n_att x image_dim, torch.Size([128, 1, 2048])
                batch_size = att_reshape.size(0)
                image_embedding = tmp_embedding.view(batch_size, -1)

                return image_embedding, attn_temp
            
            else:
                print('not implemented')

        elif attr == 'context':
            s_oo, s_ot = s_homo, s_hetero
            oo_edge_feature, ot_edge_feature = homo_edge_feature, hetero_edge_feature
            if (s_oo is not None) and (oo_edge_feature is not None) and \
                (s_ot is not None) and (ot_edge_feature is not None) and \
                (pre_ques_embed is not None):
                text_feat_variable = image_feat_variable
                batch_size, node_num, _ = text_feat_variable.shape  # torch.Size([32, 36, 2048])
                
                tt_feat_variable = self.GenEdgeFeature(False, 'text', text_feat_variable, oo_edge_feature, s_oo)
                to_feat_variable = self.GenEdgeFeature(True, 'text', text_feat_variable, ot_edge_feature, s_ot)
                p_t, p_tt, p_to = self.image_attention_model(
                    attr, text_feat_variable, ques_embed, image_dims,
                    oo_edge_feature=tt_feat_variable, s_oo=s_oo,
                    ot_edge_feature=to_feat_variable, s_ot=s_ot,
                )  # torch.Size([128, 100, 1])
                attn_weight = F.softmax(self.prior_weight_fc_3_text(pre_ques_embed.mean(1)), dim=1)
                attn_weight_t = attn_weight[:, 0].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_weight_tt = attn_weight[:, 1].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_weight_to = attn_weight[:, 2].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_temp_t = attn_weight_t * p_t + attn_weight_tt * p_tt + attn_weight_to * p_to # torch.Size([128, 100, 1])

                text_dim = text_feat_variable.shape[2] # 350
                x_text = text_feat_variable
                g_t = torch.bmm(attn_temp_t.permute(0,2,1), x_text)
                updated_ocr = attn_temp_t.repeat(1,1,text_dim)*text_feat_variable
                return g_t, updated_ocr

            elif (s_oo is not None) and (oo_edge_feature is not None) and (pre_ques_embed is not None):
                batch_size, node_num, _ = image_feat_variable.shape  # torch.Size([32, 36, 300])
                
                oo_feat_variable = self.GenTTFeature(False, 'text', image_feat_variable, oo_edge_feature,
                                                          s_oo)
 
                attention, attention_obj = self.image_attention_model(
                    attr, image_feat_variable, ques_embed, image_dims,
                    oo_edge_feature=oo_feat_variable,
                    s_oo=s_oo
                )  # torch.Size([128, 100, 1])
                attn_weight = F.softmax(self.prior_weight_fc(pre_ques_embed.mean(1)), dim=1)
                attn_weight_exp_1 = attn_weight[:, 0].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_weight_exp_2 = attn_weight[:, 1].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_temp = attn_weight_exp_1 * attention + attn_weight_exp_2 * attention_obj
                att_reshape = attn_temp.permute(0, 2, 1)  # torch.Size([128, 1, 100])

                tmp_embedding = torch.bmm(
                    att_reshape, image_feat_variable
                )  # N x n_att x image_dim, torch.Size([128, 1, 2048])
                image_dim = image_feat_variable.shape[2]
                image_feat_variable_graph = attn_temp.repeat(1,1,image_dim)*image_feat_variable
                batch_size = att_reshape.size(0)
                image_embedding = tmp_embedding.view(batch_size, -1)

                return image_embedding, attn_temp, image_feat_variable_graph

            elif (s_ot is not None) and (ot_edge_feature is not None) and (pre_ques_embed is not None):
                batch_size, node_num, _ = image_feat_variable.shape  # torch.Size([32, 36, 300])
                
                ot_feat_variable = self.GenTOFeature(True, 'text', image_feat_variable, ot_edge_feature,
                                                           s_ot)
 
                attention, attention_obj = self.image_attention_model(
                    attr, image_feat_variable, ques_embed, image_dims,
                    oo_edge_feature=ot_feat_variable,
                    s_oo=s_ot
                )  # torch.Size([128, 100, 1])
                attn_weight = F.softmax(self.prior_weight_fc(pre_ques_embed.mean(1)), dim=1)
                attn_weight_exp_1 = attn_weight[:, 0].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_weight_exp_2 = attn_weight[:, 1].unsqueeze(1).expand(batch_size, node_num).unsqueeze(2)
                attn_temp = attn_weight_exp_1 * attention + attn_weight_exp_2 * attention_obj
                # attn_temp = attention + attention_obj
                # att_reshape = attention.permute(0, 2, 1)  # torch.Size([128, 1, 100])
                att_reshape = attn_temp.permute(0, 2, 1)  # torch.Size([32, 1, 50])

                tmp_embedding = torch.bmm(
                    att_reshape, image_feat_variable
                )  # N x n_att x image_dim, torch.Size([128, 1, 2048])
                image_dim = image_feat_variable.shape[2]
                image_feat_variable_graph = attn_temp.repeat(1,1,image_dim)*image_feat_variable
                batch_size = att_reshape.size(0)
                image_embedding = tmp_embedding.view(batch_size, -1)

                return image_embedding, attn_temp, image_feat_variable_graph

    def GenEdgeFeature(self, hetero, cent, image_feat_variable, oo_edge_feature, s_oo):
        batch_size, node_num, _ = image_feat_variable.shape  # torch.Size([32, 36, 2048])
        _, _, knn, join_dim = oo_edge_feature.shape  # torch.Size([32, 36, 5, 2053])]
        oo_edge_feature_view = oo_edge_feature.view(-1, join_dim)  # (batch_size*node_num*knn, 5+2048)->torch.Size([5760, 2053])
        if hetero == False and cent == 'obj':
            oo_edge_feature = self.edge_oo_emb(oo_edge_feature_view)  # (batch_size*node_num*knn, 2048)->torch.Size([5760, 2048])
        if hetero == True and cent == 'obj':
            oo_edge_feature = self.edge_ot_emb(oo_edge_feature_view)
        if hetero == False and cent == 'text':
            oo_edge_feature = self.edge_tt_emb(oo_edge_feature_view)
        if hetero == True and cent =='text':
            oo_edge_feature = self.edge_to_emb(oo_edge_feature_view)
        # the obj-obj edge feature
        oo_edge_feature = oo_edge_feature.view(batch_size, node_num, knn,-1)  # (batch_size, node_num, knn=5, 2048)->torch.Size([32, 36, 5, 2048])
        # oo_joint_attn_list = torch.zeros((batch_size, node_num, knn, 1)).float()  # (node_num, knn=5, 1)
        # the knn-attention of edge, calculate A^(oo)
        oo_edge_feature_view_2 = oo_edge_feature.view(batch_size, node_num * knn,-1)  # torch.Size([32, 180, 2048])
        if hetero == False:
            joint_oo_feature = self.modal_combine_layer(oo_edge_feature_view_2,s_oo,attr='edge')  # torch.Size([batch_size, node_num*5, 5000])->torch.Size([32, 180, 5000])
            raw_joint_oo_attn = self.transform_layer(joint_oo_feature)  # torch.Size([batch_size, node_num*5, 1])->torch.Size([32, 180, 1])
        if hetero == True:
            joint_oo_feature = self.hetero_modal_combine_layer(oo_edge_feature_view_2,s_oo,attr='edge')
            raw_joint_oo_attn = self.hetero_transform_layer(joint_oo_feature)
        raw_joint_oo_attn_view = raw_joint_oo_attn.view(batch_size, node_num, knn,-1)  # torch.Size([batch_size, node_num, 5, 1])->torch.Size([32, 36, 5, 1])
        joint_oo_attention = nn.functional.softmax(raw_joint_oo_attn_view,dim=2)  # torch.Size([batch_size, node_num, 5, 1])->torch.Size([32, 36, 5, 1])
        joint_oo_attention = joint_oo_attention.cuda()
        oo_feat_variable = (joint_oo_attention * oo_edge_feature).sum(2)

        return oo_feat_variable

class ImageFinetune(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super(ImageFinetune, self).__init__()
        with open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3
