'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


# 继承了BaseSolver类
class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)            # 需要COCO格式的评估
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }
        best_metric = -1

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()            # 更新学习率
            
            if self.output_dir:                 # 保存模型
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(                  # 验证评估
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir, epoch
            )

            # 保存最佳模型
            current_metric = test_stats[list(test_stats.keys())[0]][0]      # 获取第一个指标的值
            if current_metric > best_metric:
                best_metric = current_metric
                if self.output_dir and dist.is_main_process():
                    best_model_path = self.output_dir / 'checkpoint_best.pth'
                    dist.save_on_master(self.state_dict(epoch), best_model_path)
                    print(f"New best model saved at epoch {epoch} with metric {current_metric:.4f}")

            # TODO 
            # for k in test_stats.keys():
            #     if k in best_stat:
            #         best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
            #         best_stat[k] = max(best_stat[k], test_stats[k][0])
            #     else:
            #         best_stat['epoch'] = epoch
            #         best_stat[k] = test_stats[k][0]
            # print('best_stat: ', best_stat)
            for k, v in test_stats.items():
                # 如果是 list 或 tuple，则取第一个值；否则直接使用
                value = v[0] if isinstance(v, (list, tuple)) else v
                if k in best_stat:
                    best_stat['epoch'] = epoch if value > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], value)
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = value
            print('best_stat: ', best_stat)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'inference_time_sec': test_stats.get('inference_time_sec', None),
                        'inference_fps': test_stats.get('inference_fps', None)}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:          # 保存COCO eval结果
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir, epoch='Unknown')
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
