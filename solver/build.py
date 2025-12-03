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
        # Gate Projection（最高优先级）
        if "gate_proj" in key:
            lr = args.lr * args.lr_factor
            # ✅ 打印不同模块的 gate
            if "visual.transformer" in key:
                print(f"🔥 Vision Gate: {key}, lr={lr:.2e}")
            elif "transformer.resblocks" in key and "cross" not in key:
                print(f"🔥 Text Gate: {key}, lr={lr:.2e}")
            elif "cross" in key:
                print(f"🔥 Cross-Modal Gate: {key}, lr={lr:.2e}")
            elif "id_gate" in key:
                print(f"🔥 ID Branch Gate: {key}, lr={lr:.2e}")
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

    print("\n=== Learning Rate Assignment Summary ===")
    print(f"Base LR: {args.lr:.2e}")
    print(f"LR Factor: {args.lr_factor}")
    print(f"Bias LR Factor: {args.bias_lr_factor}")
    print(f"High LR: {args.lr * args.lr_factor:.2e}")

    print("\n--- ID Module Assignment (including ID gates) ---")
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        # 显示所有ID相关参数（包括门控）
        if ("text_id_" in key or "image_id_" in key or 
            "id_query" in key or "id_attention" in key or 
            "id_layernorm" in key):
            actual_lr = args.lr * args.bias_lr_factor if "bias" in key else args.lr * args.lr_factor
            gate_mark = "🔥 [Gate]" if "gate_proj" in key else "✅"
            print(f"{gate_mark} LR={actual_lr:.2e}: {key}")

    print("\n--- CMT Gate Assignment ---")
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        # 只显示CMT的gate_proj（排除ID门控）
        if "gate_proj" in key and "cross_modal_transformer" in key:
            actual_lr = args.lr * args.bias_lr_factor if "bias" in key else args.lr * args.lr_factor
            print(f"🔥 LR={actual_lr:.2e}: {key}")

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
