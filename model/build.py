from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn
from collections import OrderedDict


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        # optional decoupled heads for alignment and identity tasks
        self.use_decouple = getattr(args, 'decouple', False)
        if self.use_decouple:
            def make_mlp(in_dim, out_dim, layers):
                modules = []
                hidden = out_dim
                for i in range(layers):
                    modules.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
                    if i < layers - 1:
                        modules += [QuickGELU(), LayerNorm(hidden)]
                mlp = nn.Sequential(*modules) if modules else nn.Identity()
                # 添加初始化
                for m in mlp.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.02)  # CLIP使用的标准差
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)
                    elif isinstance(m, LayerNorm):
                        nn.init.constant_(m.weight, 1.0)
                        nn.init.constant_(m.bias, 0.0)
                return mlp
            
            # 选择ID头的架构类型
            id_head_type = getattr(args, 'id_head_type', 'mlp')  # mlp, transformer, hybrid

            if id_head_type == 'transformer':
                # 使用Transformer Block作为ID头（更强的表达能力）
                print(f"Using Transformer ID head with {args.id_head_layers} layers")
                self.id_img_head = Transformer(
                    width=self.embed_dim,
                    layers=args.id_head_layers,
                    heads=self.embed_dim // 64
                )
                self.id_txt_head = Transformer(
                    width=self.embed_dim,
                    layers=args.id_head_layers,
                    heads=self.embed_dim // 64
                )
                self.use_transformer_head = True

            elif id_head_type == 'hybrid':
                # 混合架构：Self-Attention + FFN (类似Transformer Block但更轻量)
                print(f"Using Hybrid ID head")

                class HybridIDHead(nn.Module):
                    def __init__(self, embed_dim, num_heads=8):
                        super().__init__()
                        self.ln1 = LayerNorm(embed_dim)
                        self.self_attn = nn.MultiheadAttention(
                            embed_dim, num_heads, batch_first=True
                        )
                        self.ln2 = LayerNorm(embed_dim)
                        self.mlp = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4),
                            QuickGELU(),
                            nn.Linear(embed_dim * 4, embed_dim)
                        )

                    def forward(self, x):
                        # x: [batch, embed_dim]
                        x = x.unsqueeze(1)  # [batch, 1, embed_dim]
                        # Self-attention with residual
                        attn_out = self.self_attn(
                            self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False
                        )[0]
                        x = x + attn_out
                        # FFN with residual
                        x = x + self.mlp(self.ln2(x))
                        return x.squeeze(1)  # [batch, embed_dim]

                self.id_img_head = HybridIDHead(self.embed_dim)
                self.id_txt_head = HybridIDHead(self.embed_dim)
                self.use_transformer_head = False

            else:
                # 默认MLP (保持向后兼容)
                print(f"Using MLP ID head with {args.id_head_layers} layers")
                self.id_img_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)
                self.id_txt_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)
                self.use_transformer_head = False

            # # identity heads (separate per modality)
            # self.id_img_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)
            # self.id_txt_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)

            # predictor heads for reconstruction constraint - 第二个block
            id_pred_layers = getattr(args, 'id_pred_layers', 1)
            self.id_pred_img = make_mlp(self.embed_dim, self.embed_dim, id_pred_layers)
            self.id_pred_txt = make_mlp(self.embed_dim, self.embed_dim, id_pred_layers)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        i_feats = x[:, 0, :].float()
        # Revert to original inference: directly use backbone features
        return i_feats
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        t_feats = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        # Revert to original inference: directly use backbone features
        return t_feats

    def forward(self, batch):
        ret = dict()

        # print(f"Current task: {self.current_task}")
        # print(f"use_decouple: {getattr(self, 'use_decouple', False)}")
        
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        # Use original features for both alignment and ID tasks (no decoupled heads)
        ai_feats, at_feats = i_feats, t_feats

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(ai_feats, at_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(ai_feats, at_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(ai_feats, at_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            # 两段式ID分支：第一段做身份判别，第二段做重建约束
            if getattr(self, 'use_decouple', False):
                # 第一个block：backbone -> ID空间 (用于身份判别)
                if getattr(self, 'use_transformer_head', False):
                    # Transformer需要序列输入 [batch, seq_len, embed_dim]
                    id_i_feats = self.id_img_head(i_feats.unsqueeze(1).permute(1, 0, 2)).permute(1, 0, 2).squeeze(1)
                    id_t_feats = self.id_txt_head(t_feats.unsqueeze(1).permute(1, 0, 2)).permute(1, 0, 2).squeeze(1)
                else:
                    # MLP和Hybrid直接处理 [batch, embed_dim]
                    id_i_feats = self.id_img_head(i_feats)
                    id_t_feats = self.id_txt_head(t_feats)
                
                # 第二个block：ID空间 -> 重建空间 (用于语义约束)
                pred_i_feats = self.id_pred_img(id_i_feats)
                pred_t_feats = self.id_pred_txt(id_t_feats)
                
                # 重建约束：确保ID空间不偏离原始语义空间太远
                # 使用detach()确保梯度只更新ID分支，不影响骨干网络
                reg_weight = getattr(self.args, 'reg_loss_weight', 0.05)

                # 选择不同的约束方式
                reg_type = getattr(self.args, 'reg_loss_type', 'mse')  # mse, cosine, huber
                
                if reg_type == 'cosine':
                    # 余弦相似度损失（更温和）
                    reg_loss_img = 1 - torch.nn.functional.cosine_similarity(
                        pred_i_feats, i_feats.detach(), dim=-1).mean()
                    reg_loss_txt = 1 - torch.nn.functional.cosine_similarity(
                        pred_t_feats, t_feats.detach(), dim=-1).mean()
                    reg_loss = reg_loss_img + reg_loss_txt
                elif reg_type == 'huber':
                    # Huber损失（鲁棒性更好）
                    reg_loss = torch.nn.functional.huber_loss(
                        torch.nn.functional.normalize(pred_i_feats, dim=-1),
                        torch.nn.functional.normalize(i_feats.detach(), dim=-1),
                        delta=0.1
                    ) + torch.nn.functional.huber_loss(
                        torch.nn.functional.normalize(pred_t_feats, dim=-1),
                        torch.nn.functional.normalize(t_feats.detach(), dim=-1),
                        delta=0.1
                    )
                else:  # 默认MSE
                    backbone_i_norm = torch.nn.functional.normalize(i_feats.detach(), dim=-1)
                    backbone_t_norm = torch.nn.functional.normalize(t_feats.detach(), dim=-1)
                    pred_i_norm = torch.nn.functional.normalize(pred_i_feats, dim=-1)
                    pred_t_norm = torch.nn.functional.normalize(pred_t_feats, dim=-1)
                    
                    reg_loss = torch.nn.functional.mse_loss(pred_i_norm, backbone_i_norm) + \
                            torch.nn.functional.mse_loss(pred_t_norm, backbone_t_norm)
                ret.update({'reg_loss': reg_loss * reg_weight})
            else:
                # 原始方式：直接使用骨干特征
                id_i_feats, id_t_feats = i_feats, t_feats

            # ID分类损失
            image_logits = self.classifier(id_i_feats)  # 使用ID空间特征
            text_logits = self.classifier(id_t_feats)   # 使用ID空间特征
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    # convert_weights(model)
    return model
