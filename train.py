import argparse
import os
import torch
from pytorch_lightning.loggers.neptune import NeptuneLogger
import neptune
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from core.utils import load_cfg, load_weights, select_weights
from core.distiller import Distiller
from core.model_zoo import model_zoo

torch.cuda.max_memory_allocated()


def build_logger(cfg):
    # return NeptuneLogger(project="MobileStyleGAN.pytorch")
    run = NeptuneLogger(project='zhoutjcc/MobileStyleGAN-pytorch',
                        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNTUzNTlhMy0zZjgzLTQyMzktOGFlNS05MDE0ZDcxNmUyOTcifQ==")
    return run
    # return getattr(pl_loggers, cfg.type)(
    #     **cfg.params
    # )


def main(args):
    cfg = load_cfg(args.cfg)
    distiller = Distiller(cfg)
    if args.ckpt is not None:
        ckpt = model_zoo(args.ckpt)
        load_weights(distiller, ckpt["state_dict"])
    logger = build_logger(cfg.logger)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=os.getcwd() if args.checkpoint_dir is None else args.checkpoint_dir,
    #     save_top_k=True,
    #     save_last=True,
    #     verbose=True,
    #     monitor=cfg.trainer.monitor,
    #     mode=cfg.trainer.monitor_mode,
    #
    # )
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=cfg.trainer.max_epochs,
        accumulate_grad_batches=args.grad_batches,
        accelerator='gpu',
        devices=1,
        checkpoint_callback=True,
        val_check_interval=args.val_check_interval,
        logger=logger
    )
    if args.export_model is None:
        trainer.fit(distiller)
    elif args.export_model == "onnx":
        distiller.to_onnx(args.export_dir, args.export_w_plus)
    elif args.export_model == "coreml":
        distiller.to_coreml(args.export_dir, args.export_w_plus)
    else:
        raise "Unknown export format."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--gpus", type=int, default=1, help="number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default="ddp", choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint", help="path to checkpoint_dir")
    parser.add_argument("--val-check-interval", type=int, default=500, help="validation check interval")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--cfg", type=str, help="path to config file")
    parser.add_argument("--export-model", type=str, default=None, help="export model for deploy. ['onnx', 'coreml']")
    parser.add_argument("--export-dir", type=str, default="./train", help="path to export model")
    parser.add_argument("--export-w-plus", action='store_true', help="export with W+ space")
    args = parser.parse_args()
    main(args)
