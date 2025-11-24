import pdb
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.simple_tokenizer import SimpleTokenizer


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size, text_length=args.text_length)
        self.embed_dim = base_cfg['embed_dim']

        self.context_length = base_cfg['context_length']
        self.vocab_size = base_cfg['vocab_size']


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
                # print(f"Using Transformer ID head with {args.id_head_layers} layers")
                # self.id_img_head = Transformer(
                #     width=self.embed_dim,
                #     layers=args.id_head_layers,
                #     heads=self.embed_dim // 64
                # )
                # self.id_txt_head = Transformer(
                #     width=self.embed_dim,
                #     layers=args.id_head_layers,
                #     heads=self.embed_dim // 64
                # )

                print(f"Using SHARED Transformer ID head with {args.id_head_layers} layers")
                self.id_head = Transformer(
                    width=self.embed_dim,
                    layers=args.id_head_layers,
                    heads=self.embed_dim // 64
                )


                self.use_transformer_head = True

            elif id_head_type == 'hybrid':
                # 混合架构：Self-Attention + FFN (类似Transformer Block但更轻量)
                # print(f"Using Hybrid ID head")

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
                print(f"Using SHARED Hybrid ID head")
                # ... (HybridIDHead 类的定义不变) ...
                self.id_head = HybridIDHead(self.embed_dim)
                # self.id_img_head = HybridIDHead(self.embed_dim)
                # self.id_txt_head = HybridIDHead(self.embed_dim)

                self.use_transformer_head = False

            else:
                # 默认MLP (保持向后兼容)
                # print(f"Using MLP ID head with {args.id_head_layers} layers")
                # self.id_img_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)
                # self.id_txt_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)

                print(f"Using SHARED MLP ID head with {args.id_head_layers} layers")
                self.id_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)
                self.use_transformer_head = False

            # # identity heads (separate per modality)
            # self.id_img_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)
            # self.id_txt_head = make_mlp(self.embed_dim, self.embed_dim, args.id_head_layers)

            # predictor heads for reconstruction constraint - 第二个block
            id_pred_layers = getattr(args, 'id_pred_layers', 1)
            # self.id_pred_img = make_mlp(self.embed_dim, self.embed_dim, id_pred_layers)
            # self.id_pred_txt = make_mlp(self.embed_dim, self.embed_dim, id_pred_layers)
            print(f"Using SHARED Predictor head with {id_pred_layers} layers")
            self.id_predictor = make_mlp(self.embed_dim, self.embed_dim, id_pred_layers)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

            # 1. 在 __init__ 中添加新的注意力池化模块
            # 可学习的 "ID Query" 向量
            # 它将学会 "提问": "这段文本中的身份信息是什么？"
            self.text_id_query = nn.Parameter(torch.randn(1, 1, self.embed_dim)) # [1, 1, D]
            
            # 多头注意力 (MHA) 层
            # Q = text_id_query, K = text_feats, V = text_feats
            self.text_id_attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.embed_dim // 64, # 512-dim -> 8 heads
                batch_first=True
            )
            # LayerNorm 用于稳定训练
            self.text_id_layernorm = LayerNorm(self.embed_dim)

            # 1. 在 __init__ 中添加新的 图像 注意力池化模块
            # 可学习的 "ID Query" 向量 (图像侧)
            self.image_id_query = nn.Parameter(torch.randn(1, 1, self.embed_dim)) # [1, 1, D]
            # MHA (Q = image_id_query, K = image_patches, V = image_patches)
            self.image_id_attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.embed_dim // 64,
                batch_first=True
            )
            self.image_id_layernorm = LayerNorm(self.embed_dim)


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
                            ('fc', nn.Linear(self.embed_dim, self.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)



    # def _compute_masked_text_mean(self, text_feats, caption_ids):
    #     """
    #     辅助函数：计算掩码平均池化，忽略 SOT, EOT, 和 PAD
    #     text_feats: [B, 77, D]
    #     caption_ids: [B, 77]
    #     """
    #     # 1. 找到 SOT (index 0) 和 EOT (argmax)
    #     eot_indices = caption_ids.argmax(dim=-1) # [B]    
    #     # 2. 创建掩码 (1 for words, 0 for SOT/EOT/PAD)
    #     # 初始全 1
    #     mask = torch.ones_like(caption_ids, dtype=torch.float32, device=text_feats.device)
    #     # 掩掉 SOT (index 0)
    #     mask[:, 0] = 0
    #     # 掩掉 EOT 和之后所有的 PAD
    #     for i in range(caption_ids.shape[0]):
    #         mask[i, eot_indices[i]:] = 0
        
    #     # 3. 计算掩码平均
    #     mask_unsqueezed = mask.unsqueeze(-1) # [B, 77, 1]
        
    #     # 逐元素相乘，非内容词变为 0
    #     masked_feats = text_feats * mask_unsqueezed
        
    #     # 求和
    #     sum_feats = torch.sum(masked_feats, dim=1) # [B, D]
        
    #     # 计算每个句子的内容词数量
    #     num_words = torch.sum(mask, dim=1).unsqueeze(-1) # [B, 1]
        
    #     # 避免除以 0 (如果某个句子没有内容词)
    #     num_words = num_words.clamp(min=1e-6)
        
    #     # 求平均
    #     mean_feats = sum_feats / num_words
        
    #     return mean_feats.float()

    def _compute_attention_pooled_text_id(self, text_feats, caption_ids):
        """
        使用可学习的 [ID_QUERY] + 多头注意力，从文本中池化身份特征。
        忽略 SOT(位置0)、EOT 以及 EOT 之后的 PAD。
        text_feats: [B, 77, D]
        caption_ids: [B, 77]
        """
        # 1) 计算每个样本的 EOT 位置
        eot_indices = caption_ids.argmax(dim=-1) # [B]
        B, L = caption_ids.shape
        # 2) 向量化构造 key_padding_mask (True=Masked, False=Not Masked)
        # 只保留 [1, eot) 的内容词，SOT(0)、EOT以及 EOT 之后(含 PAD) 全部 Mask
        pos = torch.arange(L, device=caption_ids.device).unsqueeze(0).expand(B, L)  # [B, L]
        content_tokens = (pos >= 1) & (pos < eot_indices.unsqueeze(1))  # [B, L] bool
        mask = ~content_tokens  # True->mask, False->keep

        # 3) 准备 Query: [1,1,D] -> [B,1,D]
        query = self.text_id_query.expand(text_feats.size(0), -1, -1)

        # 4) MHA (Q=query, K/V=text_feats)
        attn_output, _ = self.text_id_attention(
            query,
            text_feats,
            text_feats,
            key_padding_mask=mask  # [B,L] bool
        )  # [B,1,D]

        # 5) 后处理
        id_feat = attn_output.squeeze(1)  # [B,D]
        id_feat = self.text_id_layernorm(id_feat)
        return id_feat.float()

    def _compute_attention_pooled_image_id(self, image_feats):
        """
        辅助函数：使用可学习的 [ID_QUERY] token 和多头注意力
        来池化图像的 Patch 特征 (忽略 CLS token)。
        image_feats: [B, N_patches + 1, D] (index 0 是 CLS)
        """
        # 1. 准备 Query
        query = self.image_id_query.expand(image_feats.shape[0], -1, -1) # [B, 1, D]

        # 2. 准备 Key 和 Value (K=V)
        # 我们只使用 patch (从 index 1 开始)，忽略 CLS token
        patches = image_feats[:, 1:, :] # [B, N_patches, D]

        # 3. MHA (Q=query, K=patches, V=patches)
        # 注意：这里我们不需要 key_padding_mask，因为所有 patch 都是有效的
        attn_output, _ = self.image_id_attention(
            query, 
            patches, 
            patches, 
            key_padding_mask=None
        ) # [B, 1, D]
        
        # 4. 后处理
        id_feat = attn_output.squeeze(1) # [B, D]
        id_feat = self.image_id_layernorm(id_feat)
        
        return id_feat.float()


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
        # 获取所有 token 特征
        # x: [B, num_patches + 1, D] index0 是 CLS token
        x = self.base_model.encode_image(image)

        i_feats_inst = x[:, 0, :].float() # [CLS]
        # 提取实例和 ID 特征
        # i_feats_inst = x[:, 0, :].float() # [CLS_Inst]
        # i_feats_id = x[:, 1, :].float()   # [CLS_ID]
        
        # 身份 (ID) 特征: Patch tokens 平均池化
        # i_feats_id = torch.mean(x[:, 1:, :], dim=1).float()
        i_feats_id = self._compute_attention_pooled_image_id(x)


        # 根据评估模式返回特征
        mode = getattr(self.args, 'inference_fusion', 'align')
        
        if mode == 'id':
            # 只使用 ID 特征
            return F.normalize(i_feats_id, dim=-1) # ID模式也应返回归一化特征
        elif mode == 'fuse':
            # 融合实例和 ID 特征
            # alpha = getattr(self.args, 'fusion_alpha', 0.5)
            # (融合策略可以有很多种，这里用归一化后加权)
            # return F.normalize(alpha * F.normalize(i_feats_inst, dim=-1) + \
                            #  (1 - alpha) * F.normalize(i_feats_id, dim=-1), dim=-1)
            # 将两个特征相加，然后归一化
            # return F.normalize(i_feats_inst + i_feats_id, dim=-1)
            # 1. 单独归一化
            i_feats_inst_norm = F.normalize(i_feats_inst, dim=-1)
            i_feats_id_norm = F.normalize(i_feats_id, dim=-1)
            # 2. 拼接
            fused_feat = torch.cat([i_feats_inst_norm, i_feats_id_norm], dim=1) # [B, D*2]
            # 3. 归一化拼接后的向量
            return F.normalize(fused_feat, dim=-1)
        else: # 默认 'align'
            return F.normalize(i_feats_inst, dim=-1) # Align模式也应返回归一化特征
        
        
        
        # Revert to original inference: directly use backbone features
        # return i_feats
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        # x: [B, 77, D]
        x = self.base_model.encode_text(text)
        # t_feats = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        eot_indices = text.argmax(dim=-1)
        t_feats_eot = x[torch.arange(x.shape[0]), eot_indices].float() # [EOT]
        # t_feats_sot = x[:, 0, :].float()  # [SOT]
        # id_t_feats_mapped = t_feats_sot

        # 身份 (ID) 特征: Word tokens 掩码平均池化
        # t_feats_sot = self._compute_masked_text_mean(x, text)
        t_feats_sot = self._compute_attention_pooled_text_id(x, text)
        id_t_feats_mapped = t_feats_sot

        # 4. 根据评估模式返回特征
        mode = getattr(self.args, 'inference_fusion', 'align')
        
        if mode == 'id':
            # 只使用 ID 特征
            return F.normalize(id_t_feats_mapped, dim=-1) # ID模式也应返回归一化特征
        elif mode == 'fuse':
            # 融合实例和 ID 特征
            # alpha = getattr(self.args, 'fusion_alpha', 0.5)
            # return F.normalize(alpha * F.normalize(t_feats_eot, dim=-1) + \
            #                  (1 - alpha) * F.normalize(id_t_feats_mapped, dim=-1), dim=-1)
            # 将两个特征相加，然后归一化
            # return F.normalize(t_feats_eot + id_t_feats_mapped, dim=-1)
            # 1. 单独归一化
            t_feats_eot_norm = F.normalize(t_feats_eot, dim=-1)
            id_t_feats_mapped_norm = F.normalize(id_t_feats_mapped, dim=-1)
            # 2. 拼接
            fused_feat = torch.cat([t_feats_eot_norm, id_t_feats_mapped_norm], dim=1) # [B, D*2]
            # 3. 归一化拼接后的向量
            return F.normalize(fused_feat, dim=-1)
        else: # 默认 'align'
            # 只使用实例特征 (你当前的行为)
            return F.normalize(t_feats_eot, dim=-1) # Align模式也应返回归一化特征


        # Revert to original inference: directly use backbone features
        # return t_feats

    def forward(self, batch):
        ret = dict()

        # print(f"Current task: {self.current_task}")
        # print(f"use_decouple: {getattr(self, 'use_decouple', False)}")
        # 骨干网络特征提取
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)# [Batch, Num_Patches + 1, Embed_Dim]以及[Batch, Seq_Len, Embed_Dim]
        
    
        ai_feats = image_feats[:, 0, :].float() # Image[CLS_Inst] (index 0)

        # 找到 EOT token
        # caption_ids 是 [SOT, w1, ..., EOT, 0]
        # text_feats 索引与 caption_ids 是一一对应的。
        eot_indices = caption_ids.argmax(dim=-1)
        at_feats = text_feats[torch.arange(text_feats.shape[0]), eot_indices].float() # Text[EOT]
        
        # 2. 提取身份特征 (Identity Features) - "i_feats_id", "t_feats_id"
        # i_feats_id = image_feats[:, 1, :].float() # Image[CLS_ID] (index 1)
        # t_feats_id = text_feats[:, 0, :].float()  # Text[SOT] (index 0)
        # 2. 提取身份特征 (Identity Features)
        #    - 图像: 使用 Patch tokens 平均池化
        #    - 文本: 使用 Word tokens 掩码平均池化
        # i_feats_id = torch.mean(image_feats[:, 1:, :], dim=1).float() # Image[Patches] mean, 从索引1开始到最后的所有 token
        i_feats_id = self._compute_attention_pooled_image_id(image_feats)
        # t_feats_id = self._compute_masked_text_mean(text_feats, caption_ids) # Text[Words] mean
        t_feats_id = self._compute_attention_pooled_text_id(text_feats, caption_ids)
        


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
                    id_i_feats = self.id_head(i_feats_id.unsqueeze(1).permute(1, 0, 2)).permute(1, 0, 2).squeeze(1)
                    id_t_feats = self.id_head(t_feats_id.unsqueeze(1).permute(1, 0, 2)).permute(1, 0, 2).squeeze(1)
                else:
                    # MLP和Hybrid直接处理 [batch, embed_dim]
                    id_i_feats = self.id_head(i_feats_id)
                    id_t_feats = self.id_head(t_feats_id)
                
                # 第二个block：ID空间 -> 重建空间 (用于语义约束)
                # ! 现在的重建目标应该是原始的ID特征（i_feats_id&t_feats_id）
                pred_i_feats = self.id_predictor(id_i_feats)
                pred_t_feats = self.id_predictor(id_t_feats)
                
                # 重建约束：确保ID空间不偏离原始语义空间太远
                # 使用detach()确保梯度只更新ID分支，不影响骨干网络
                reg_weight = getattr(self.args, 'reg_loss_weight', 0.05)

                # 选择不同的约束方式
                reg_type = getattr(self.args, 'reg_loss_type', 'mse')  # mse, cosine, huber
                
                if reg_type == 'cosine':
                    # 余弦相似度损失（更温和）
                    reg_loss_img = 1 - torch.nn.functional.cosine_similarity(
                        pred_i_feats, i_feats_id.detach(), dim=-1).mean()
                    reg_loss_txt = 1 - torch.nn.functional.cosine_similarity(
                        pred_t_feats, t_feats_id.detach(), dim=-1).mean()
                    reg_loss = reg_loss_img + reg_loss_txt
                elif reg_type == 'huber':
                    # Huber损失（鲁棒性更好）
                    reg_loss = torch.nn.functional.huber_loss(
                        torch.nn.functional.normalize(pred_i_feats, dim=-1),
                        torch.nn.functional.normalize(i_feats_id.detach(), dim=-1),
                        delta=0.1
                    ) + torch.nn.functional.huber_loss(
                        torch.nn.functional.normalize(pred_t_feats, dim=-1),
                        torch.nn.functional.normalize(t_feats_id.detach(), dim=-1),
                        delta=0.1
                    )
                else:  # 默认MSE
                    backbone_i_norm = torch.nn.functional.normalize(i_feats_id.detach(), dim=-1)
                    backbone_t_norm = torch.nn.functional.normalize(t_feats_id.detach(), dim=-1)
                    pred_i_norm = torch.nn.functional.normalize(pred_i_feats, dim=-1)
                    pred_t_norm = torch.nn.functional.normalize(pred_t_feats, dim=-1)
                    
                    reg_loss = torch.nn.functional.mse_loss(pred_i_norm, backbone_i_norm) + \
                            torch.nn.functional.mse_loss(pred_t_norm, backbone_t_norm)
                ret.update({'reg_loss': reg_loss * reg_weight})
            else:
                # 原始方式：直接使用骨干特征
                # id_i_feats, id_t_feats = i_feats_id, t_feats_id
                id_i_feats = i_feats_id  # 使用 Image[Patches] mean
                id_t_feats = t_feats_id    # 使用 Text[Words] mean


            # 强制归一化，让两者在同一个超球面上公平竞争
            # id_i_feats = F.normalize(id_i_feats, p=2, dim=-1)
            # id_t_feats = F.normalize(id_t_feats, p=2, dim=-1)
            # ID分类损失
            image_logits = self.classifier(id_i_feats)  # 使用ID空间特征
            text_logits = self.classifier(id_t_feats)   # 使用ID空间特征

            id_temperature = getattr(self.args, 'id_temperature', 2.0)
            id_label_smoothing = getattr(self.args, 'id_label_smoothing', 0.1)


            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'], temperature=id_temperature, label_smoothing=id_label_smoothing)*self.args.id_loss_weight})

            if 'triplet' in self.current_task: # 检查是否在 loss_names 中启用了 triplet
                triplet_loss = objectives.compute_triplet(
                    id_i_feats, 
                    id_t_feats, 
                    batch['pids'],
                    margin=self.args.triplet_margin
                )
                ret.update({'triplet_loss': triplet_loss * self.args.triplet_loss_weight})



            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            # mlm_ids = batch['mlm_ids'] # [B, 77] (SOT, w1..w73, EOT, 0, 0)

            # pdb.set_trace()

            # # mlm_feats [B, 77, D] 与 mlm_ids 索引对应
            # mlm_feats = self.base_model.encode_text(mlm_ids)

            # x = self.cross_former(mlm_feats, image_feats, image_feats)

            # x = self.mlm_head(x)  # [batch_size, 77, vocab_size]

            # scores = x.float().reshape(-1, self.vocab_size) # [B*77, vocab_size]

            # # 1. 原始 labels [B, 77]
            # # (来自修改后的 bases.py, -100 是 ignore index)
            # mlm_labels = batch['mlm_labels'] 
            
            # # 2. 展平 (特征和标签已对齐)
            # mlm_labels_aligned = mlm_labels.reshape(-1) # [B*77]
            
            # # 3. (旧的错误逻辑被移除，不再需要 torch.cat 和 ignore_labels)

            # # 4. 使用对齐后的 labels 计算 loss (假设 objectives.compute_mlm 使用 ignore_index=-100)
            # ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels_aligned)*self.args.mlm_loss_weight})
            # # =================

            # # === 修正 ACC 计算 ===
            # pred = scores.max(1)[1] # [B*77]
            
            # # 确保只在 "非-100" 的真实标签上计算准确率
            # mlm_label_idx = torch.nonzero(mlm_labels_aligned != -100).squeeze()
            
            # if mlm_label_idx.numel() > 0: # 避免没有mask token时出错
            #     acc = (pred[mlm_label_idx] == mlm_labels_aligned[mlm_label_idx]).float().mean()
            # else:
            #     acc = torch.tensor(0.0, device=scores.device)
            
            # ret.update({'mlm_acc': acc})
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
