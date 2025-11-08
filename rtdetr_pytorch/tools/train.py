"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))      # 动态添加模块搜索路径
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()

    # tuning和resume不能同时为true
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(               # ./src/core/yaml_config.py
        args.config,                # ./configs/rtdetr/rtdetr_r18vd_6x_coco.yml
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning,
        offset_pretrain=args.offset_pretrain,
        freeze_offset=args.freeze_offset
    )

    # ./rtdetr/include/rtdetr_r50vd中task:detection，读取任务类型
    # TASKS['detection']就是DetectionSolver，是一个字符串到类的映射
    # 等同于DetectionSolver(cfg)，构造对应任务的solver实例-->进入det_solver类
    solver = TASKS[cfg.yaml_cfg['task']](cfg)           # 创建子类solver， ./src/solver/solver.py+det_solver.py
    
    if args.test_only:          # 验证流程
        solver.val()
    else:                       # 调用子类中实现的训练逻辑
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', '-c', type=str, default="../configs/rtdetr/rtdetr_r18vd_6x_coco.yml")
    parser.add_argument('--config', '-c', type=str, default="../configs/rtdetr/rtdetr_dual.yml")
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--offset_pretrain', '-o', type=str, )
    parser.add_argument('--freeze_offset', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)
