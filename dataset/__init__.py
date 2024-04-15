# Build Dataset
from torch.utils.data import DataLoader
#from .bigbench import HFBigBenchDataset, CoTCollator
from configs_data._dataset_config import _C as DATASET_CFG
import importlib
from torch.utils.data import ConcatDataset
from copy import deepcopy
class CoTCollator():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, batch):
        _ = list(zip(*batch))
        batch = {
            "indexes": _[0],
            "prompts" : _[1],
            "answers" : _[2],
            "responses": _[3] if len(_) > 3 else None,
        }
        return batch

def build_dataset(cfg, data_config_name, split, is_train, datapool_path = None, override_cfg = None):
    dataset_cfg = DATASET_CFG.clone()
    dataset_cfg.merge_from_file(data_config_name)
    if override_cfg is not None:
        print('Overriding dataset config with {}'.format(override_cfg))
        dataset_cfg.merge_from_file(override_cfg)

    # auto import
    module_name = dataset_cfg.MODULE_NAME
    class_name = dataset_cfg.CLASS_NAME
    module = importlib.import_module(module_name)
    dataset_class = getattr(module, class_name)

    dataset = dataset_class(
        cfg = cfg, 
        split = split, 
        dataset_cfg = dataset_cfg,
        datapool_path = datapool_path,
        is_train = is_train)
    return dataset

def build_dataloader(cfg, split, is_train=True):
    
    # will try to load DATASET_CFG
    datasets = []
    #zip  cfg.TRAIN.DATAPOOL_PATH.split(",") and cfg.DATA.CONFIG.split(",")
    dataset_files = cfg.DATA.CONFIG.split(",")
    datapool_paths = cfg.TRAIN.DATAPOOL_PATH.split(",")
    if cfg.DATA.OVERRIDE_CONFIG is not None:
        if "," in cfg.DATA.OVERRIDE_CONFIG:
            override_cfgs = cfg.DATA.OVERRIDE_CONFIG.split(",")
        else:
            override_cfgs = [cfg.DATA.OVERRIDE_CONFIG] * len(dataset_files)
    if len(cfg.DATA.COPIES) == 0:
        copies = [1] * len(dataset_files)
    else:
        copies = cfg.DATA.COPIES

    for dataset_file, datapool_path, override_cfg, copy in zip(dataset_files, datapool_paths, override_cfgs, copies):       
        dataset = build_dataset(
            cfg, dataset_file, split, 
            is_train = is_train,
            datapool_path = datapool_path,
            override_cfg=override_cfg)
        datasets.append(dataset)
        for i in range(copy - 1):
            datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    print("Concate dataset size {}, split: {}".format(len(dataset), split))
    for i in dataset.datasets:
        print("  Dataset size {}, HF name: {}".format(len(i), i.dataset_cfg.HF_IDENTIFIER))

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.DATA.BATCH_SIZE if is_train else cfg.DATA.EVAL_BATCH_SIZE,
        shuffle=is_train and cfg.DATA.SHUFFLE_TRAIN, 
        drop_last=is_train,
        collate_fn=CoTCollator(cfg),
        num_workers=cfg.DATA.NUM_WORKERS)
    return dataloader