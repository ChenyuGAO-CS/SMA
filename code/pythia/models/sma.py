# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import torch
import math
from torch import nn
import torch.nn.functional as F
import os

from pythia.utils.general import get_pythia_root

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from pythia.modules.layers import ClassifierLayer
from pythia.modules.encoders import ImageEncoder
from pythia.modules.embeddings import ImageEmbedding

@registry.register_model("sma")
class SMA(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self.mmt_config = BertConfig(**self.config.mmt)
        self.mmt = MMT(self.mmt_config)
        self.so_to_mmt_in = nn.Linear(3*1536, self.mmt_config.hidden_size)
        self.st_to_mmt_in = nn.Linear(3*1536, self.mmt_config.hidden_size)
        self.so_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.st_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.so_drop = nn.Dropout(0.1)
        self.st_drop = nn.Dropout(0.1)
        self.linear_go_to_mmt_in = nn.Linear(2048, self.mmt_config.hidden_size)
        self.linear_gt_to_mmt_in = nn.Linear(300, self.mmt_config.hidden_size)
        self.go_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.gt_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.go_drop = nn.Dropout(0.1)
        self.gt_drop = nn.Dropout(0.1)
        self.linear_updated_ocr_to_mmt_in = nn.Linear(300, self.mmt_config.hidden_size)
        self.updated_ocr_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.updated_ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)
        self.linear_joint = nn.Linear(1536,768)
        self.answer_processor =  registry.get(self._datasets[0] + "_answer_processor")
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._init_text_embeddings("text")
        # init feature embedding for "image"
        setattr(self, "image_feature_dim", self.config["image_feature_dim"])
        self.feature_embeddings_out_dim = 0
        feature_attn_model_params = self.config["image_feature_embeddings"][0]
        feature_embedding = ImageEmbedding(
            getattr(self, "image_feature_dim"),
            self.text_embeddings_out_dim,
            **feature_attn_model_params
        )
        self.feature_embeddings_out_dim += feature_embedding.out_dim
        self.feature_embeddings_out_dim *= getattr(self, "image_feature_dim")
        setattr(
            self, "image_feature_embeddings_out_dim", self.feature_embeddings_out_dim
        )
        del self.feature_embeddings_out_dim
        setattr(
            self,
            "image_feature_embedding",
            feature_embedding
        )
        # init feature embedding for "context"
        setattr(self, "context_feature_dim", self.config["context_feature_dim"])
        self.feature_embeddings_out_dim = 0
        feature_attn_model_params = self.config["context_feature_embeddings"][0]
        feature_embedding = ImageEmbedding(
            getattr(self, "context_feature_dim"),
            self.text_embeddings_out_dim,
            **feature_attn_model_params
        )
        self.feature_embeddings_out_dim += feature_embedding.out_dim
        self.feature_embeddings_out_dim *= getattr(self, "context_feature_dim")
        setattr(
            self, "context_feature_embeddings_out_dim", self.feature_embeddings_out_dim
        )
        del self.feature_embeddings_out_dim
        setattr(
            self,
            "context_feature_embedding",
            feature_embedding
        )
        self._init_combine_layer("image", "text")
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=768,
            out_dim=num_choices-50,
            **self.config["classifier"]["params"]
        )

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            pythia_root = get_pythia_root()
            self.text_bert = TextBert.from_pretrained(
                os.path.join(pythia_root, self.config.model_data_dir,'bert'), config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append({
                'module': self.text_bert,
                'lr_scale': self.config.lr_scale_text_bert,
            })
        else:
            self.writer.write('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

    def _build_obj_encoding(self):
        self.obj_dim = 2048
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append({
            'module': self.obj_faster_rcnn_fc7,
            'lr_scale': 0.1,
        })
        # OBJ location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.obj_dim
        )
        self.obj_feat_layer_norm = nn.LayerNorm(self.obj_dim)
        self.obj_bbox_layer_norm = nn.LayerNorm(self.obj_dim)
        self.obj_drop = nn.Dropout(0.1)

    def _build_ocr_encoding(self):
        self.ocr_fastext_dim = 300
        self.ocr_phoc_dim = 604
        self.ocr_RCNN_dim = 2048
        self.transformer_cnn_dim = 512
        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        self.finetune_modules.append({
            'module': self.ocr_faster_rcnn_fc7,
            'lr_scale': 0.1,
        })

        # OCR appearance feature: relative Fasttext + PHOC + FasterRCNN
        self.linear_ocr_appear_to_mmt_in = nn.Linear(
            self.ocr_fastext_dim+self.ocr_RCNN_dim+self.ocr_phoc_dim+self.transformer_cnn_dim, self.ocr_fastext_dim
            # self.ocr_fastext_dim+self.ocr_RCNN_dim+self.ocr_phoc_dim, self.ocr_fastext_dim
        )
        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.ocr_fastext_dim
        )
        self.ocr_feat_layer_norm = nn.LayerNorm(self.ocr_fastext_dim)
        self.ocr_bbox_layer_norm = nn.LayerNorm(self.ocr_fastext_dim)
        self.ocr_drop = nn.Dropout(0.1)

    def _forward_obj_encoding(self, sample_list):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0[:,:36,:]
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        obj_bbox = sample_list.obj_bbox[:,:36]
        obj_mmt_in = (
            self.obj_feat_layer_norm(
                obj_feat
            ) + self.obj_bbox_layer_norm(
                self.linear_obj_bbox_to_mmt_in(obj_bbox)
            )
        )
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        return obj_mmt_in

    def _forward_ocr_encoding(self, sample_list):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_phoc
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:,:ocr_fasttext.size(1),:]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)
        assert ocr_fc7.size(-1) == 2048

        # OCR appearance feature: Transformer global representation feature
        ocr_trans = sample_list.image_feature_2[:,:ocr_fasttext.size(1),:]
        ocr_trans = F.normalize(ocr_trans, dim=-1)
        assert ocr_trans.size(-1) == 512

        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_fc7, ocr_phoc, ocr_trans],
            # [ocr_fasttext, ocr_fc7, ocr_phoc],
            dim=-1
        )
        
        ocr_bbox = sample_list.ocr_bbox.coordinates
        ocr_mmt_in = (
                    self.ocr_feat_layer_norm(
                        self.linear_ocr_appear_to_mmt_in(ocr_feat)
                    ) + self.ocr_bbox_layer_norm(
                        self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
                    )
        )
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        return ocr_mmt_in

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * m['lr_scale']
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

    def forward(self, sample_list):
        txt_inds = sample_list.text
        txt_mask = _get_mask(sample_list.text_len, sample_list.text.size(1))
        text_bert_out = self.text_bert(txt_inds=txt_inds,txt_mask=txt_mask)
        sample_list.text = text_bert_out

        _, s_o, s_oo, s_ot, s_t, s_tt, s_to = self.process_text_embedding(sample_list)

        obj_encoded_feats = self._forward_obj_encoding(sample_list)
        ocr_encoded_feats = self._forward_ocr_encoding(sample_list)
        g_o = self.process_feature_embedding(
            "image", sample_list, s_o, s_homo=s_oo, s_hetero=s_ot, 
            pre_ques_embed=sample_list.text, obj_feats=obj_encoded_feats, ocr_feats=ocr_encoded_feats
        )
        g_t, updated_ocr = self.process_feature_embedding(
            "context", sample_list, s_t, s_homo=s_tt, s_hetero=s_to, 
            pre_ques_embed=sample_list.text, obj_feats=obj_encoded_feats, ocr_feats=ocr_encoded_feats
        )  # torch.Size([128, 350])

        s_o = torch.cat((s_o,s_oo,s_ot),dim=-1)
        s_t = torch.cat((s_t,s_tt,s_to),dim=-1)
        s_o = self.so_drop(self.so_layer_norm(self.so_to_mmt_in(s_o.unsqueeze(1)) ))
        s_t = self.st_drop(self.st_layer_norm(self.st_to_mmt_in(s_t.unsqueeze(1)) ))
        so_mask = torch.ones(s_o.size(0),s_o.size(1),dtype=torch.float32,device=s_o.device)
        st_mask = torch.ones(s_t.size(0),s_t.size(1),dtype=torch.float32,device=s_t.device)        
        g_o = self.go_drop(self.go_layer_norm(self.linear_go_to_mmt_in(g_o)))
        g_t = self.gt_drop(self.gt_layer_norm(self.linear_gt_to_mmt_in(g_t)))
        go_mask = torch.ones(g_o.size(0),g_o.size(1),dtype=torch.float32,device=g_o.device)
        gt_mask = torch.ones(g_t.size(0),g_t.size(1),dtype=torch.float32,device=g_t.device)

        ocr_emb = self.updated_ocr_drop(self.updated_ocr_layer_norm(self.linear_updated_ocr_to_mmt_in(updated_ocr)))
        ocr_tokens = sample_list.context
        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        ocr_mask = _get_mask(ocr_nums, ocr_tokens.size(1))
       
        if self.training:
            prev_inds = sample_list.train_prev_inds.clone()
            mmt_results = self.mmt(
                s_o, so_mask, s_t, st_mask, 
                g_o, go_mask, g_t, gt_mask, 
                ocr_emb=ocr_emb,
                ocr_mask=ocr_mask,
                fixed_ans_emb=self.classifier.module.weight,
                prev_inds=prev_inds,
            )
            g_O = mmt_results["mmt_so_output"]*mmt_results["mmt_go_output"]
            g_T = mmt_results["mmt_st_output"]*mmt_results["mmt_gt_output"]
            update_joint_embedding = torch.cat((g_O, g_T),dim=-1) # torch.Size([32, 1, 1536])
            update_joint_embedding = self.linear_joint(update_joint_embedding)
            mmt_dec_output = mmt_results["mmt_dec_output"] # torch.Size([32, 12, 768])
            score_feature = torch.cat([update_joint_embedding, mmt_dec_output[:,1:,:]], dim=-2)
            mmt_ocr_output = mmt_results["mmt_ocr_output"] 
            fixed_scores = self.classifier(score_feature)
            dynamic_ocr_scores = self.ocr_ptr_net(
                score_feature, mmt_ocr_output, ocr_mask
            )
            scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            prev_inds = torch.zeros_like(
                sample_list.train_prev_inds
            )
            prev_inds[:, 0] = self.answer_processor.BOS_IDX
            
            # greedy decoding at test time
            for t in range(dec_step_num):
                mmt_results = self.mmt(
                    s_o, so_mask, s_t, st_mask, 
                    g_o, go_mask, g_t, gt_mask, 
                    ocr_emb=ocr_emb,
                    ocr_mask=ocr_mask,
                    fixed_ans_emb=self.classifier.module.weight,
                    prev_inds=prev_inds,
                )
                if t==0:
                    g_O = mmt_results["mmt_so_output"]*mmt_results["mmt_go_output"]
                    g_T = mmt_results["mmt_st_output"]*mmt_results["mmt_gt_output"]
                    update_joint_embedding = torch.cat((g_O, g_T),dim=-1) # torch.Size([32, 1, 1536])
                    update_joint_embedding = self.linear_joint(update_joint_embedding)
                mmt_dec_output = mmt_results["mmt_dec_output"]
                score_feature = torch.cat([update_joint_embedding, mmt_dec_output[:,1:,:]], dim=-2)
                mmt_ocr_output = mmt_results["mmt_ocr_output"]
                fixed_scores = self.classifier(score_feature)
                dynamic_ocr_scores = self.ocr_ptr_net(
                    score_feature, mmt_ocr_output, ocr_mask
                )
                scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = scores.argmax(dim=-1)
                prev_inds[:, 1:] = argmax_inds[:, :-1]

        return {"scores": scores} 

class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output

class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self,
                so,so_mask,st,st_mask,
                go,go_mask,gt,gt_mask,
                ocr_emb,
                ocr_mask,
                fixed_ans_emb,
                prev_inds):
        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)
        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(dec_emb.size(0),dec_emb.size(1),dtype=torch.float32,device=dec_emb.device)
        encoder_inputs = torch.cat(
            [so,st,go,gt,ocr_emb,dec_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [so_mask,st_mask,go_mask,gt_mask,ocr_mask,dec_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        so_max_num = so_mask.size(-1)
        st_max_num = st_mask.size(-1)
        go_max_num = go_mask.size(-1)
        gt_max_num = gt_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        so_begin = 0
        so_end = so_max_num
        st_begin = so_max_num
        st_end = st_begin + st_max_num
        go_begin = so_max_num + st_max_num 
        go_end = go_begin + go_max_num
        gt_begin = so_max_num  + st_max_num  + go_max_num 
        gt_end = gt_begin + gt_max_num
        ocr_begin = so_max_num + st_max_num + go_max_num + gt_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
            _get_causal_mask(dec_max_num, encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_so_output = mmt_seq_output[:, so_begin:so_end]
        mmt_st_output = mmt_seq_output[:, st_begin:st_end]
        mmt_go_output = mmt_seq_output[:, go_begin:go_end]
        mmt_gt_output = mmt_seq_output[:, gt_begin:gt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_so_output': mmt_so_output,
            'mmt_st_output': mmt_st_output,
            'mmt_go_output': mmt_go_output,
            'mmt_gt_output': mmt_gt_output,
            'mmt_ocr_output': mmt_ocr_output,
            'mmt_dec_output': mmt_dec_output,
        }
        return results

class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores

class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=ocr_emb.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb

def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask

@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask

def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results