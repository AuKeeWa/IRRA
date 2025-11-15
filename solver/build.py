import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        # 跨模态模块 (cross_attn, cross_modal_transformer)
        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        # 分类器、MLM头、ID相关模块
        # if "classifier" in key or "mlm_head" in key or "text_id_" in key or "image_id_" in key or "fusion_" in key:
        #     lr = args.lr * args.lr_factor
        if ("classifier" in key or 
            "mlm_head" in key or 
            "text_id_" in key or 
            "image_id_" in key or 
            "id_pooling" in key or      # 多层池化模块
            "id_query" in key or        # 单层Query（方案1/2）
            "id_attention" in key or    # 共享MHA（方案1/2）
            "id_layernorm" in key or    # 共享LN（方案1/2）
            "shared_id_" in key or      # 显式共享命名
            "fusion_" in key):          # 特征融合
            lr = args.lr * args.lr_factor
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    print("\n=== ID Module Learning Rate Assignment ===")
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        
        # 检查是否匹配高学习率规则
        is_high_lr = ("classifier" in key or 
                    "mlm_head" in key or 
                    "text_id_" in key or 
                    "image_id_" in key or 
                    "id_pooling" in key or 
                    "id_query" in key or 
                    "id_attention" in key or 
                    "id_layernorm" in key or 
                    "shared_id_" in key or 
                    "fusion_" in key)
        
        if is_high_lr and ("id_" in key or "pooling" in key):
            lr = args.lr * args.lr_factor
            print(f"✅ High LR ({lr:.2e}): {key}")

    print("=" * 50)

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
