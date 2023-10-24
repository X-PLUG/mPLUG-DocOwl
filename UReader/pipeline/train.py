import argparse
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from sconf import Config
from icecream import ic
from peft import LoraConfig, get_peft_config, get_peft_model
from transformers import Trainer
from transformers.training_args import TrainingArguments

from mplug_owl import MplugOwlForConditionalGeneration, MplugOwlTokenizer
from pipeline.data_utils import train_valid_test_datasets_provider
from pipeline.utils import batchify, set_args
from pipeline.trainer import CustomTrainer
from pipeline.utils import add_config_args

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--pretrained-ckpt', type=str, default='MAGAer13/mplug-owl-llama-7b-pt',
                    help='Path to the pretrained checkpoint.')
parser.add_argument('--inference_mode', type=bool, default=False,
                    help='The inference mode.')
parser.add_argument('--seq-length', type=int, default=1024,
                    help='Maximum sequence length to process.')
# parser.add_argument('--use-lora', action='store_true', help='LORA.')
parser.add_argument('--freeze-v2t', action='store_true', help='Freeze abstractor')
parser.add_argument('--language-training-method', type=str, default='lora', help='LORA.')
parser.add_argument('--lora-r', type=int, default=8,
                    help='curvature.')
parser.add_argument('--lora-alpha', type=int, default=32,
                    help='The initialization coefficient of lora-alpha.')  
parser.add_argument('--lora-dropout', type=int, default=0.05,
                    help='The initialization coefficient of lora_dropout.')
parser.add_argument('--bf16', action='store_true', default=False,
                    help='Run model in bfloat16 mode.')

# Data
parser.add_argument('--mm-config', type=str, default='configs/sft/release.yaml', help='Multimodal Config.')
parser.add_argument('--image-root', type=str, default='ureader_images', help='Image folder.')
parser.add_argument('--num-workers', type=int, default=8,
                    help="Dataloader number of workers.")  

# Training HyperParameters
parser.add_argument('--train-epochs', type=int, default=3,
                    help='Total number of epochs to train over all '
                    'training runs.')
parser.add_argument('--micro-batch-size', type=int, default=None,
                    help='Batch size per model instance (local batch size). '
                    'Global batch size is local batch size times data '
                    'parallel size times number of micro batches.')
parser.add_argument('--global-batch-size', type=int, default=256,
                    help='Batch size per model instance (local batch size). '
                    'Global batch size is local batch size times data '
                    'parallel size times number of micro batches.')
parser.add_argument('--lr', type=float, default=None,
                    help='Initial learning rate. Depending on decay style '
                    'and initial warmup, the learing rate at each '
                    'iteration would be different.')
parser.add_argument('--min-lr', type=float, default=1e-6,
                    help='Minumum value for learning rate. The scheduler'
                    'clip values below this threshold.')
parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='Weight decay coefficient for L2 regularization.')
parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                    help='The gradient accumulation steps. If the global and micro batch size are given, this variable will be computed automatically.')
parser.add_argument('--clip-grad', type=float, default=1.0,
                    help='Gradient clipping based on global L2 norm.')
parser.add_argument('--adam-beta1', type=float, default=0.9,
                    help='First coefficient for computing running averages '
                    'of gradient and its square')
parser.add_argument('--adam-beta2', type=float, default=0.999,
                    help='Second coefficient for computing running averages '
                    'of gradient and its square')
parser.add_argument('--adam-eps', type=float, default=1e-08,
                    help='Term added to the denominator to improve'
                    'numerical stability')

parser.add_argument('--num-warmup-steps', type=int, default=50,
                    help='The number of warmup steps.')

# Evaluation & Save
parser.add_argument('--save-path', type=str, default=None,
                    help='Output directory to save checkpoints to.')
parser.add_argument('--save-interval', type=int, default=None,
                    help='Number of iterations between checkpoint saves.')
parser.add_argument('--eval-iters', type=int, default=100,
                    help='Number of iterations to run for evaluation'
                    'validation/test for.')

# Other
parser.add_argument('--gradient-checkpointing', action='store_true',
                    help='The gradient checkpointing.')
parser.add_argument('--logging-nan-inf-filter', action='store_true',
                    help='The logging nan inf filter.')
parser.add_argument('--ddp-find-unused-parameters', action='store_true',
                    help='unused parameters finding.')
parser.add_argument('--do-train', action='store_true', default=True,
                    help='Whether to do training.')  
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank')

parser.add_argument('--tensorboard-dir', type=str)
parser.add_argument('--deepspeed', type=str, default=None)

def get_accumulation_step(args):
    global_batch_size = args.global_batch_size
    batch_size = args.micro_batch_size
    gpu_nums = dist.get_world_size()

    accumulation_step = max(1,int(round(global_batch_size/(batch_size*gpu_nums))))
    if accumulation_step*(batch_size*gpu_nums) != global_batch_size:
        import warnings
        warnings.warn(f"The actual global_batch_size is {accumulation_step*(batch_size*gpu_nums)} instead {global_batch_size}\n")
    return accumulation_step
    
def main():
    args, left_argv = parser.parse_known_args()  
    torch.distributed.init_process_group(backend="nccl")
    ic(left_argv)
    config = Config(args.mm_config)
    add_config_args(config, args)
    # args.patch_pos_embed_type = config.get('patch_pos_embed_type', 'post')
    # ic(args.patch_pos_embed_type)
    if args.global_batch_size is not None:
        args.gradient_accumulation_steps = get_accumulation_step(args)
    ic(args.gradient_accumulation_steps)
    set_args(args)

    model = MplugOwlForConditionalGeneration.from_pretrained(
        args.pretrained_ckpt,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float,
    )
    # if not args.bf16:
    #     model = model.half()
    tokenizer = MplugOwlTokenizer.from_pretrained(args.pretrained_ckpt)


    for name, param in model.named_parameters():
        if 'vision_model' in name:
            # 默认vision不训练
            param.requires_grad = False
        elif 'language_model' in name:
            # 下面根据language状态进行修改
            param.requires_grad = False
        else:
            if args.freeze_v2t and ('query_tokens' in name or 'abstractor' in name):
                # 如果freeze则不训练 默认打开
                param.requires_grad = False
                continue
            param.requires_grad = True

    if args.language_training_method == 'lora':
        peft_config = LoraConfig(
            # target_modules=r'.*language_model.*\.(q_proj|v_proj)', 
            target_modules=r'.*\.(q_proj|v_proj)', 
            inference_mode=args.inference_mode, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        model.language_model = get_peft_model(model.language_model, peft_config)
        model.language_model.print_trainable_parameters()
    elif args.language_training_method == 'training':
        for param in model.parameters():
            if 'language_model' in name:
                param.requires_grad = True
    else:
        pass

    if args.gradient_checkpointing:
        # abs do not use gradient checkpointing
        # set vit gradient checkpointing
        model.vision_model.apply(
            partial(model.vision_model._set_gradient_checkpointing, value=True))
        ic(model.vision_model.encoder.gradient_checkpointing)
        # set llama gradient checkpointing
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.language_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.language_model.apply(
            partial(model.language_model._set_gradient_checkpointing, value=True))

    model.train()

    train_data, valid_data = train_valid_test_datasets_provider(
        config.data_files, config=config, 
        tokenizer=tokenizer, seq_length=args.seq_length,
        image_root=args.image_root,
    )
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=TrainingArguments(
            learning_rate=args.lr,
            warmup_steps=args.num_warmup_steps,
            do_train=args.do_train,
            num_train_epochs=args.train_epochs,
            output_dir=args.save_path,
            logging_dir=args.tensorboard_dir,
            save_strategy='steps',
            save_steps=args.save_interval,
            evaluation_strategy='steps',
            eval_steps=args.eval_iters,
            per_device_train_batch_size=args.micro_batch_size,
            max_grad_norm=args.clip_grad,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            fp16=not args.bf16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=args.eval_iters//4,
            logging_nan_inf_filter=args.logging_nan_inf_filter,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            deepspeed=args.deepspeed,
            dataloader_num_workers=args.num_workers
        ),
    )


    trainer.train()

    model.save_pretrained(args.save_path)

if __name__ == '__main__':
    main()