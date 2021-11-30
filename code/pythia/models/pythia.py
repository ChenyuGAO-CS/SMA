# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.embeddings import (ImageEmbedding, PreExtractedEmbedding,
                                       TextEmbedding)
from pythia.modules.encoders import ImageEncoder
from pythia.modules.layers import (ClassifierLayer, ModalCombineLayer,
                                   ReLUWithWeightNormFC)
from pythia.utils.configuration import ConfigNode

from pythia.utils.graph_construction import gen_intra_edge_feature
from pythia.utils.graph_construction import gen_extra_edge_feature
from pythia.utils.graph_construction import find_k_nearest_node
from pythia.utils.graph_construction import hetero_find_k_nearest_node

@registry.register_model("pythia")
class Pythia(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    def build(self):
        self._build_word_embedding()
        self._init_text_embeddings("text")
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")
        self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _build_word_embedding(self):
        assert len(self._datasets) > 0
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)

    def _init_text_embeddings(self, attr="text"):
        if "embeddings" not in attr:
            attr += "_embeddings"

        text_embeddings = []
        text_embeddings_list_config = self.config[attr]

        embeddings_out_dim = 0

        for text_embedding in text_embeddings_list_config:
            embedding_type = text_embedding.type
            embedding_kwargs = ConfigNode(text_embedding.params)

            self._update_text_embedding_args(embedding_kwargs)

            embedding = TextEmbedding(embedding_type, **embedding_kwargs)

            text_embeddings.append(embedding)
            embeddings_out_dim += embedding.text_out_dim

        setattr(self, attr + "_out_dim", embeddings_out_dim)
        setattr(self, attr, nn.ModuleList(text_embeddings))

    def _update_text_embedding_args(self, args):
        # Add model_data_dir to kwargs
        args["model_data_dir"] = self.config["model_data_dir"]

    def _init_feature_encoders(self, attr):
        feat_encoders = []
        feat_encoders_list_config = self.config[attr + "_feature_encodings"]
        feature_dim = self.config[attr + "_feature_dim"]
        setattr(self, attr + "_feature_dim", feature_dim)

        for feat_encoder in feat_encoders_list_config:
            encoder_type = feat_encoder["type"]
            encoder_kwargs = feat_encoder["params"]
            encoder_kwargs["model_data_dir"] = self.config["model_data_dir"]

            feat_model = ImageEncoder(encoder_type, feature_dim, **encoder_kwargs)

            feat_encoders.append(feat_model)
            setattr(self, attr + "_feature_dim", feat_model.out_dim)

        setattr(self, attr + "_feature_encoders", nn.ModuleList(feat_encoders))

    def _init_feature_embeddings(self, attr):
        feature_embeddings_list = []
        num_feature_feat = len(
            getattr(self.config, "{}_feature_encodings".format(attr))
        )

        self.feature_embeddings_out_dim = 0

        for _ in range(num_feature_feat):
            feature_embeddings = []
            feature_attn_model_list = self.config[attr + "_feature_embeddings"]

            for feature_attn_model_params in feature_attn_model_list:
                feature_embedding = ImageEmbedding(
                    getattr(self, attr + "_feature_dim"),
                    self.text_embeddings_out_dim,
                    **feature_attn_model_params
                )
                feature_embeddings.append(feature_embedding)
                self.feature_embeddings_out_dim += feature_embedding.out_dim

            feature_embeddings = nn.ModuleList(feature_embeddings)
            feature_embeddings_list.append(feature_embeddings)

        self.feature_embeddings_out_dim *= getattr(self, attr + "_feature_dim")

        setattr(
            self, attr + "_feature_embeddings_out_dim", self.feature_embeddings_out_dim
        )
        del self.feature_embeddings_out_dim
        setattr(
            self,
            attr + "_feature_embeddings_list",
            nn.ModuleList(feature_embeddings_list),
        )

    def _get_embeddings_attr(self, attr):
        embedding_attr1 = attr
        if hasattr(self, attr + "_embeddings_out_dim"):
            embedding_attr1 = attr + "_embeddings_out_dim"
        else:
            embedding_attr1 = attr + "_feature_embeddings_out_dim"

        return embedding_attr1

    def _init_combine_layer(self, attr1, attr2):
        config_attr = attr1 + "_" + attr2 + "_modal_combine"

        multi_modal_combine_layer = ModalCombineLayer(
            self.config[config_attr]["type"],
            getattr(self, self._get_embeddings_attr(attr1)),
            getattr(self, self._get_embeddings_attr(attr2)),
            **self.config[config_attr]["params"]
        )

        setattr(
            self,
            attr1 + "_" + attr2 + "_multi_modal_combine_layer",
            multi_modal_combine_layer,
        )

    def _init_classifier(self, combined_embedding_dim):
        # TODO: Later support multihead
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")

        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=combined_embedding_dim,
            out_dim=num_choices,
            **self.config["classifier"]["params"]
        )

    def _init_extras(self):
        self.inter_model = None

    def get_optimizer_parameters(self, config):
        combine_layer = self.image_text_multi_modal_combine_layer
        params = [
            {"params": self.word_embedding.parameters()},
            {"params": self.image_feature_embeddings_list.parameters()},
            {"params": self.text_embeddings.parameters()},
            {"params": combine_layer.parameters()},
            {"params": self.classifier.parameters()},
            {
                "params": self.image_feature_encoders.parameters(),
                "lr": (config["optimizer_attributes"]["params"]["lr"] * 0.1),
            },
        ]

        return params

    def _get_classifier_input_dim(self):
        return self.image_text_multi_modal_combine_layer.out_dim

    def process_text_embedding(
        self, sample_list, embedding_attr="text_embeddings", info=None
    ):
        text_embeddings = []

        # Get "text" attribute in case of "text_embeddings" case
        # and "context" attribute in case of "context_embeddings"
        texts = getattr(sample_list, embedding_attr.split("_")[0])

        # Get embedding models
        text_embedding_models = getattr(self, embedding_attr)

        for text_embedding_model in text_embedding_models:
            # TODO: Move this logic inside
            if isinstance(text_embedding_model, PreExtractedEmbedding):
                embedding = text_embedding_model(sample_list.question_id)
            else:
                embedding = text_embedding_model(texts)
            text_embeddings.append(embedding)

        # # visualize decomposed question attention
        # image_id = getattr(sample_list, "image_id")
        # question_id = getattr(sample_list, "question_id").cpu()
        # question_id = question_id.numpy()
        # batch_size_t, _, _ = text_embeddings[0][7].shape
        # for cnt in range(0, batch_size_t):
        #     # image_path_org = './save/temp_check/'+question_id[cnt]+'image_id.pdh'
        #     # torch.save(image_id[cnt], image_path_org)
        #     attn_path_org = './save/temp_check/'+str(question_id[cnt])+'_a_o.pdh'
        #     torch.save(text_embeddings[0][7][cnt], attn_path_org)
        #     attn_path_org = './save/temp_check/'+str(question_id[cnt])+'_a_oo.pdh'
        #     torch.save(text_embeddings[0][8][cnt], attn_path_org)
        #     attn_path_org = './save/temp_check/'+str(question_id[cnt])+'_a_ot.pdh'
        #     torch.save(text_embeddings[0][9][cnt], attn_path_org)
        #     attn_path_org = './save/temp_check/'+str(question_id[cnt])+'_a_t.pdh'
        #     torch.save(text_embeddings[0][10][cnt], attn_path_org)
        #     attn_path_org = './save/temp_check/'+str(question_id[cnt])+'_a_tt.pdh'
        #     torch.save(text_embeddings[0][11][cnt], attn_path_org)
        #     attn_path_org = './save/temp_check/'+str(question_id[cnt])+'_a_to.pdh'
        #     torch.save(text_embeddings[0][12][cnt], attn_path_org)
        return text_embeddings[0][0], text_embeddings[0][1], text_embeddings[0][2], text_embeddings[0][3], text_embeddings[0][4], text_embeddings[0][5], text_embeddings[0][6]

    def process_feature_embedding(
        self, attr, sample_list, s_central, 
        s_homo=None, s_hetero=None, pre_ques_embed=None,
        obj_feats=None, ocr_feats=None
    ):
        """
        parameters:

        input: 
        attr: "image" or "context"
        sample_list: just sample_list
        s_central: question features for guiding purpose, torch.Size([128, 2048])
                   s_o/s_t
        s_homo: s_oo/s_tt
        s_hetero: s_ot/s_to

        output:
        """
        # add obj bbox feats and image size        
        batch, bbox_num, obj_feat_dim = obj_feats.shape
        _, _, ocr_feat_dim = ocr_feats.shape
        knn_k = 5
        loc_dim = 5
        # expand obj_feats
        temp_expand_obj_feat = obj_feats[0][0]
        temp_expand_obj_feat = temp_expand_obj_feat.expand(batch,1,obj_feat_dim)*0
        temp_expand_obj_feat = torch.cat((obj_feats,temp_expand_obj_feat),1)
                    
        # expand ocr_feats
        temp_expand_ocr_feat = ocr_feats[0][0]
        temp_expand_ocr_feat = temp_expand_ocr_feat.expand(batch,1,ocr_feat_dim)*0
        temp_expand_ocr_feat = torch.cat((ocr_feats,temp_expand_ocr_feat),1)
       
        if attr == 'image':
            batch_size_t = ( sample_list.get_batch_size() )
            # Get "image_feature_0"
            feature = getattr(
                sample_list, "{}_feature_{:d}".format(attr, 0), None
            )
            feature = feature[:batch_size_t]
            # Get info related to the current feature. info is generally
            # in key of format "image_info_0" for 0th feature
            feature_info = getattr(sample_list, "{}_info_{:d}".format(attr, 0), {})
            # For Pythia, we need max_features to mask attention
            feature_dim = getattr(feature_info, "max_features", None)
            if feature_dim is not None:
                feature_dim = feature_dim[:batch_size_t]
            # Get feature embedding
            feature_embedding_model = getattr(self, attr + "_feature_embedding")
            encoded_feature = obj_feats
            batch, bbox_num, obj_feat_dim = encoded_feature.shape

            # obj_obj_edge_feature = None
            # oo edge generation
            obj_obj_edge_feature = torch.zeros((batch, bbox_num, knn_k, obj_feat_dim+loc_dim)).float()
            obj_obj_edge_feature = obj_obj_edge_feature.cuda()
            oo_edge = getattr(getattr(sample_list, "ocr_bbox"), "edge_oo")
            oo_edgefeats = getattr(getattr(sample_list, "ocr_bbox"), "edge_oofeats")
            for i in range (batch):
                obj_obj_edge_feature[i] = torch.cat((oo_edgefeats[i], temp_expand_obj_feat[i][oo_edge[i]]),2)
            
            # obj_text_edge_feature = None
            # ot edge generation
            obj_text_edge_feature = torch.zeros((batch, bbox_num, knn_k, ocr_feat_dim+loc_dim)).float()
            obj_text_edge_feature = obj_text_edge_feature.cuda()
            ot_edge = getattr(getattr(sample_list, "ocr_bbox"), "edge_ot")
            ot_edgefeats = getattr(getattr(sample_list, "ocr_bbox"), "edge_otfeats")
            for i in range (batch):
                obj_text_edge_feature[i] = torch.cat((ot_edgefeats[i], temp_expand_ocr_feat[i][ot_edge[i]]),2)

            oo_edge_feature = obj_obj_edge_feature
            ot_edge_feature = obj_text_edge_feature
            
            s_o, s_oo, s_ot = s_central, s_homo, s_hetero
            # for ablation study purpose, 
            # o feature + oo relation + ot relation
            if (s_oo is not None) and (oo_edge_feature is not None) and (s_ot is not None) and (ot_edge_feature is not None) and (pre_ques_embed is not None):
                inp = (attr, encoded_feature, s_o, feature_dim, s_oo, oo_edge_feature, s_ot, ot_edge_feature,pre_ques_embed)
            # o feature + oo relation
            elif (s_oo is not None) and (oo_edge_feature is not None) and (pre_ques_embed is not None):
                inp = (attr, encoded_feature, s_o, feature_dim, s_oo, oo_edge_feature, pre_ques_embed)
            # o feature + ot relation
            elif (s_ot is not None) and (ot_edge_feature is not None) and (pre_ques_embed is not None):
                inp = (attr, encoded_feature, s_o, feature_dim, s_ot, ot_edge_feature,pre_ques_embed)
            # o feature only
            else: inp = (attr, encoded_feature, s_o, feature_dim)
            
            g_o = feature_embedding_model(*inp)
            return g_o

        elif attr == 'context':
            batch_size_t = ( sample_list.get_batch_size() )
            # Get "context_feature_0"
            feature = getattr(
                sample_list, "{}_feature_{:d}".format(attr, 0), None
            )
            feature = feature[:batch_size_t]
            # Get info related to the current feature. info is generally
            # in key of format "image_info_0" for 0th feature
            feature_info = getattr(sample_list, "{}_info_{:d}".format(attr, 0), {})
            # For Pythia, we need max_features to mask attention
            feature_dim = getattr(feature_info, "max_features", None)
            if feature_dim is not None:
                feature_dim = feature_dim[:batch_size_t]
            # Get feature embedding
            feature_embedding_model = getattr(self, "context_feature_embedding")
            encoded_feature = ocr_feats
            batch, bbox_num, _ = encoded_feature.shape
            
            # text_text_edge_feature = None
            # tt edge generation
            text_text_edge_feature = torch.zeros((batch, bbox_num, knn_k, ocr_feat_dim+loc_dim)).float()
            text_text_edge_feature = text_text_edge_feature.cuda()
            tt_edge = getattr(getattr(sample_list, "ocr_bbox"), "edge_tt")
            tt_edgefeats = getattr(getattr(sample_list, "ocr_bbox"), "edge_ttfeats")
            for i in range (batch):
                text_text_edge_feature[i] = torch.cat((tt_edgefeats[i], temp_expand_ocr_feat[i][tt_edge[i]]),2)

            # text_obj_edge_feature = None
            # to edge generation
            text_obj_edge_feature = torch.zeros((batch, bbox_num, knn_k, obj_feat_dim+loc_dim)).float()
            text_obj_edge_feature = text_obj_edge_feature.cuda()
            to_edge = getattr(getattr(sample_list, "ocr_bbox"), "edge_to")
            to_edgefeats = getattr(getattr(sample_list, "ocr_bbox"), "edge_tofeats")
            for i in range (batch):
                text_obj_edge_feature[i] = torch.cat((to_edgefeats[i], temp_expand_obj_feat[i][to_edge[i]]),2)

            tt_edge_feature = text_text_edge_feature
            to_edge_feature = text_obj_edge_feature
            
            s_t, s_tt, s_to = s_central, s_homo, s_hetero
            # for ablation study purpose
            # t feature + tt relation + to relation
            if (s_tt is not None) and (tt_edge_feature is not None) and (s_to is not None) and (to_edge_feature is not None) and (pre_ques_embed is not None):
                inp = (attr, encoded_feature, s_t, feature_dim, s_tt, tt_edge_feature, s_to, to_edge_feature,pre_ques_embed)
            # t feature + tt relation
            elif (s_tt is not None) and (tt_edge_feature is not None) and (pre_ques_embed is not None):
                inp = (attr, encoded_feature, s_t, feature_dim, s_tt, tt_edge_feature, pre_ques_embed)
            # t feature + to relation
            elif (s_to is not None) and (to_edge_feature is not None) and (pre_ques_embed is not None):
                inp = (attr, encoded_feature, s_t, feature_dim, s_to, to_edge_feature,pre_ques_embed)
            # t feature only
            else:
                inp = (attr, encoded_feature, s_t, feature_dim)

            g_t, updated_ocr = feature_embedding_model(*inp)
            return g_t, updated_ocr

    def combine_embeddings(self, *args):
        feature_names = args[0]
        feature_embeddings = args[1]

        layer = "_".join(feature_names) + "_multi_modal_combine_layer"
        return getattr(self, layer)(*feature_embeddings)

    def calculate_logits(self, joint_embedding, **kwargs):
        return self.classifier(joint_embedding)

    def forward(self, sample_list):
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            ["image", "text"], [image_embedding_total, text_embedding_total]
        )

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        return model_output


# TODO: Update
@registry.register_model("pythia_question_only")
class PythiaQuestionOnly(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, sample_list):
        text_embedding_total = self.process_text_embedding(sample_list)
        text_embedding_total = text_embedding_total.new_zeros(
            text_embedding_total.size()
        )

        fa_txt = self.image_text_multi_modal_combine_layer.module.fa_txt
        dropout = self.image_text_multi_modal_combine_layer.module.dropout

        joint_embedding = dropout(fa_txt(text_embedding_total))

        linear_text = self.classifier.module.linear_text
        f_o_text = self.classifier.module.f_o_text
        scores = linear_text(f_o_text(joint_embedding))

        model_output = {"scores": scores}

        return model_output


# TODO: Update
@registry.register_model("pythia_image_only")
class PythiaImageOnly(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, sample_list):
        text_embedding_total = self.process_text_embedding(sample_list)
        text_embedding_total = text_embedding_total.new_zeros(
            text_embedding_total.size()
        )

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        fa_image = self.image_text_multi_modal_combine_layer.module.fa_image
        dropout = self.image_text_multi_modal_combine_layer.module.dropout

        joint_embedding = dropout(fa_image(image_embedding_total))

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        return model_output
