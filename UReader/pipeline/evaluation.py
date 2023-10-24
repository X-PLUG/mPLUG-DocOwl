import itertools
import os
import traceback
import json
import torch
from pathlib import Path
from pipeline.utils import add_config_args, set_args
import argparse
from sconf import Config
from mplug_owl.processing_mplug_owl import MplugOwlProcessor
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.configuration_mplug_owl import MplugOwlConfig
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
import torch
from pipeline.data_utils.processors.builder import build_processors
from pipeline.data_utils.processors import *
from transformers.models.llama.tokenization_llama import LlamaTokenizer

class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, jsonl):
        with open(jsonl,'r',encoding="utf-8")as f:
            self.lines = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

def collate_fn(batches):
    return batches

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_checkpoint', type=str, default=None,
                    help='Path to the trained checkpoint. If given, evaluate the given weights instead of the one in hf model.')
    parser.add_argument('--hf_model', type=str, default='./checkpoints/ureader',
                    help='Path to the huggingface model')
    args = parser.parse_args()
    config = Config('configs/sft/release.yaml')
    add_config_args(config, args)
    set_args(args)
    if not os.environ.get('MASTER_ADDR',None):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '54141'
        os.environ['LOCAL_RANK'] = '0'
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

   

    image_processor = build_processors(config['valid_processors'])['sft']

    if args.eval_checkpoint:
        state_dict = torch.load(args.eval_checkpoint)
    else:
        state_dict = None

    tokenizer = LlamaTokenizer.from_pretrained(args.hf_model)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    model = MplugOwlForConditionalGeneration.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float,
        state_dict=state_dict,
    )

    model.half()
    model.cuda()
    model.eval()
    for input_file in [
        './ureader_json/test_DeepForm.jsonl',
        './ureader_json/test_DocVQA.jsonl',
        './ureader_json/test_InfographicsVQA.jsonl',
        './ureader_json/test_KleisterCharity.jsonl',
        './ureader_json/test_TabFact.jsonl',
        './ureader_json/test_VisualMRC.jsonl',
        './ureader_json/test_WikiTableQuestions.jsonl',
        './ureader_json/test_ChartQA.jsonl',
        './ureader_json/test_TextCaps.jsonl',
        './ureader_json/test_TextVQA.jsonl',
    ]:
        dataset = InferenceDataset(input_file)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        evaluate_output=[]

        for bi,batch in enumerate(dataloader):
            for i, line in enumerate(batch):
                from PIL import Image
                
                question = line['conversations'][1]['value']
                answer = line['conversations'][2]['value']
                
                images = [Image.open(f"./ureader_images/{line['image'][0]}").convert('RGB')]
                prompt = f'Human: <image>\nHuman: {question}\nAI: '
                inputs = processor(text=prompt, images=images, return_tensors='pt')
                inputs = {k:v.cuda()for k,v in inputs.items()}
                try:
                    res = model.generate(**inputs, top_k=1)
                    model_answer = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
                except Exception as e:
                    model_answer=''
                    print(traceback.format_exc())
                line['model_answer'] = model_answer
                evaluate_output.append(line)


                print(question)
                print('gt:',answer)
                print('model:',model_answer)
                print('-'*10)
            print(f'### {bi}/{len(dataloader)} ###')
        
        torch.distributed.barrier()
        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, evaluate_output)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
        if torch.distributed.get_rank() == 0:
        
            save_path = Path(f'evaluate_results/{Path(input_file).name}')
            save_path.parent.mkdir(exist_ok=True,parents=True)

            with open(save_path,'w')as f:
                for line in merged_outputs:
                    f.write(json.dumps(line)+'\n')
        torch.distributed.barrier()  