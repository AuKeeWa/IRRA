import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "triplet_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "reg_loss": AverageMeter(), 
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    # best_top1 = 0.0
    best_r1 = 0.0
    best_mode = None

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['triplet_loss'].update(ret.get('triplet_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
            meters['reg_loss'].update(ret.get('reg_loss', 0), batch_size)  # ← 添加这一行

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                # if args.distributed:
                #     top1 = evaluator.eval(model.module.eval())
                # else:
                #     top1 = evaluator.eval(model.eval())
                # torch.cuda.empty_cache()
                # if best_top1 < top1:
                #     best_top1 = top1
                #     arguments["epoch"] = epoch
                #     checkpointer.save("best", **arguments)
                # 1. 确定要评估的正确模型实例 (处理DDP)
                model_to_eval = model.module if args.distributed else model
                
                # 2. 保存原始的融合模式，以便最后恢复
                original_fusion_mode = model_to_eval.args.inference_fusion
                
                all_results = {}
                model.eval() # 切换到评估模式

                # 3. 循环测试所有三种模式
                for mode in ['align', 'id', 'fuse']:
                    logger.info(f"--- Evaluating with mode: {mode} ---")
                    
                    # 关键：在调用 eval 之前设置模型的融合模式
                    model_to_eval.args.inference_fusion = mode
                    
                    # 假设 evaluator.eval 返回 (r1, mAP, mINP)
                    # 我们传递 model_to_eval (它已经被 model.eval() 设置为评估模式)
                    r1, mAP, mINP = evaluator.eval(model_to_eval)
                    all_results[mode] = {'R1': r1, 'mAP': mAP, 'mINP': mINP}

                # 4. 恢复原始模式 (好习惯)
                model_to_eval.args.inference_fusion = original_fusion_mode

                # 5. 打印清晰的总结表格
                logger.info("--- Validation Summary - Epoch: {} ---".format(epoch))
                table = PrettyTable(['Mode', 'R1', 'mAP', 'mINP'])
                table.add_row([
                    'Align', 
                    f"{all_results['align']['R1']:.3f}", 
                    f"{all_results['align']['mAP']:.3f}", 
                    f"{all_results['align']['mINP']:.3f}"
                ])
                table.add_row([
                    'ID', 
                    f"{all_results['id']['R1']:.3f}", 
                    f"{all_results['id']['mAP']:.3f}", 
                    f"{all_results['id']['mINP']:.3f}"
                ])
                table.add_row([
                    'Fuse', 
                    f"{all_results['fuse']['R1']:.3f}", 
                    f"{all_results['fuse']['mAP']:.3f}", 
                    f"{all_results['fuse']['mINP']:.3f}"
                ])
                logger.info("\n" + str(table))
                
                # 6. 将所有结果写入 TensorBoard
                tb_writer.add_scalar('R1/align', all_results['align']['R1'], epoch)
                tb_writer.add_scalar('mAP/align', all_results['align']['mAP'], epoch)
                tb_writer.add_scalar('R1/id', all_results['id']['R1'], epoch)
                tb_writer.add_scalar('mAP/id', all_results['id']['mAP'], epoch)
                tb_writer.add_scalar('R1/fuse', all_results['fuse']['R1'], epoch)
                tb_writer.add_scalar('mAP/fuse', all_results['fuse']['mAP'], epoch)

                torch.cuda.empty_cache()

                # 7. 根据 'fuse' 分数来保存最佳模型
                current_r1_align = all_results['align']['R1']
                current_r1_fuse = all_results['fuse']['R1']

                # 找出当前epoch中最好的模式
                if current_r1_align >= current_r1_fuse:
                    current_best_r1 = current_r1_align
                    current_best_mode = 'align'
                else:
                    current_best_r1 = current_r1_fuse
                    current_best_mode = 'fuse'


                # 与历史最佳比较
                if best_r1 < current_best_r1:
                    best_r1 = current_best_r1
                    best_mode = current_best_mode
                    arguments["epoch"] = epoch
                    logger.info(f"*** New Best R1: {best_r1:.3f} (mode: {best_mode}). Saving best.pth ***")
                    checkpointer.save("best", **arguments)

                



                
    if get_rank() == 0:
        # logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")
        logger.info(f"best R1: {best_r1} (mode: {best_mode}) at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    # top1 = evaluator.eval(model.eval())
    r1, mAP, mINP = evaluator.eval(model.eval())
    top1 = r1
    return top1
