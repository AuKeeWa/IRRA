import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))
    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    # return ce(scores, labels)
    return F.cross_entropy(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels, temperature=1.0, label_smoothing=0.1):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    if temperature != 1.0:
        image_logits = image_logits / temperature
        text_logits = text_logits / temperature

    criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss



def compute_triplet(image_features, text_features, labels, margin=0.3):
    """
    (推荐) 矢量化 (Vectorized) 的跨模态 Batch-Hard Triplet Loss
    
    Args:
        image_features: Image embeddings [batch_size, embed_dim]
        text_features: Text embeddings [batch_size, embed_dim]
        labels: Identity labels [batch_size]
        margin: Margin for triplet loss (default: 0.3)
    
    Returns:
        triplet_loss: Combined image-to-text and text-to-image triplet loss
    """
    # 归一化特征
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)
    
    # 计算成对的余弦距离矩阵
    # (距离 = 1 - 相似度), 范围 [0, 2]
    # 距离越小 = 越相似
    i2t_dist = 1 - image_norm @ text_norm.t()  # [B, B]
    t2i_dist = 1 - text_norm @ image_norm.t()  # [B, B]
    
    # --- 构造掩码 ---
    labels = labels.reshape(-1, 1)
    # pos_mask[i, j] = 1 (如果 i 和 j 是相同 ID)
    pos_mask = (labels == labels.t()).float()
    # neg_mask[i, j] = 1 (如果 i 和 j 是不同 ID)
    neg_mask = (labels != labels.t()).float()
    
    # --- 1. Image-to-Text Triplet Loss (Anchor=Image, P/N=Text) ---
    
    # 找到最难的正样本 (Hardest Positive)
    # 我们想要距离最远 (max) 的正样本
    # 将负样本的距离设置为一个非常小的值，使其在 max() 中被忽略
    hardest_pos_i2t = torch.max(i2t_dist * pos_mask, dim=1)[0] # [B]
    
    # 找到最难的负样本 (Hardest Negative)
    # 我们想要距离最近 (min) 的负样本
    # 将正样本的距离设置为一个非常大的值，使其在 min() 中被忽略
    hardest_neg_i2t = torch.min(i2t_dist + 1e5 * pos_mask, dim=1)[0] # [B]
    
    # 计算损失
    i2t_triplet_loss = torch.clamp(hardest_pos_i2t - hardest_neg_i2t + margin, min=0)
    
    # --- 2. Text-to-Image Triplet Loss (Anchor=Text, P/N=Image) ---
    
    # 找到最难的正样本 (Hardest Positive)
    hardest_pos_t2i = torch.max(t2i_dist * pos_mask, dim=1)[0] # [B]
    
    # 找到最难的负样本 (Hardest Negative)
    hardest_neg_t2i = torch.min(t2i_dist + 1e5 * pos_mask, dim=1)[0] # [B]
    
    # 计算损失
    t2i_triplet_loss = torch.clamp(hardest_pos_t2i - hardest_neg_t2i + margin, min=0)
    
    # --- 3. 合并损失 ---
    # 对所有有效的（即损失 > 0）三元组取平均
    triplet_loss = (i2t_triplet_loss + t2i_triplet_loss) / 2.0
    
    # (可选，但推荐) 只对 > 0 的损失取平均
    valid_triplets = torch.sum((triplet_loss > 1e-7).float())
    if valid_triplets > 0:
        triplet_loss = torch.sum(triplet_loss) / valid_triplets
    else:
        triplet_loss = torch.mean(triplet_loss) # 保持梯度图
        
    return triplet_loss