from einops import rearrange, repeat
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import random
from torchvision.ops.boxes import box_area
from pipeline.data_utils.randaugment import RandomAugment
from .builder import PROCESSORS
from torchvision.transforms.transforms import InterpolationMode
from torchvision.transforms import functional as F
 

def box_iou(boxes1, area1, boxes2, eps=1e-5):
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union+eps)
    return iou, union

def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    # anchors x1 y1 x2 y2

    # image_size: (h, w)
    # xyxy
    input_image_bbox = torch.tensor([0, 0, input_image_size[1], input_image_size[0]]).unsqueeze(0)

    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.clone()
    # y2
    boxes3[:,3] = input_image_size[0]/input_image_size[1]*anchors[:,2] # 用于算分辨率无关的iou
    
    area1 = anchors_areas
    
    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = shape_iou.diag()
    index = torch.argmax(shape_iou*100+iou,dim=0)
    return index

class AnchorResize(torch.nn.Module):
  
    def __init__(self, image_size, anchors, interpolation=InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        # xyxy
        self.anchors = torch.tensor(
            [[0, 0, _[1]*image_size[1], _[0]*image_size[0]] 
            for _ in anchors], requires_grad=False
        )
        
        self.anchor_areas = box_area(self.anchors)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img, skip_resize=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        selected_anchor = anchor_rank(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        target_size = self.anchors[selected_anchor][2:].tolist() # w,h
        if skip_resize:
            # for debug
            return selected_anchor
        return F.resize(img, [target_size[1],target_size[0]], self.interpolation, max_size=None, antialias=self.antialias), selected_anchor

    def __repr__(self) -> str:
        detail = f"(size={self.image_size}, anchor={self.anchors}, interpolation={self.interpolation.value}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"

@PROCESSORS.register_module()
class DocPretrainProcessor:
    def __init__(self, image_size=224, anchors=[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1),
        (1,5),(5,1),
        (1,6),(6,1),(2,3),(3,2),
        (1,7),(7,1),
        (4,2),(2,4),(1,8),(8,1),
        (3,3),(1,9),(9,1)]):
  
        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        # h,w
        self.anchors = [tuple(_) for _ in anchors]
        self.anchor_max = max([max(_) for _ in self.anchors])
        # xywh -> xyxy
        self.resizer = AnchorResize(image_size=image_size, anchors=anchors, interpolation=InterpolationMode.BICUBIC)
        self.old_resizer = transforms.Resize(image_size,interpolation=Image.BICUBIC)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.text_transform = None
        self.ocr_instructions = ['The picture reads %s.', 
                                    'The image says %s.',
                                    'there are words %s in the image.',
                                    'Words %s are in the picture.',
                                    'The texts in this image read %s.', 
                                    'The words on this picture are %s.',
                                    'The script depicted in this image reads %s.',
                                    'The writing on this visual representation states %s.',
                                    'The content presented in this diagram states %s.',
                                    'The language used in this photograph says %s.',
                                    'The inscription on this picture explains %s.',
                                    'The verbiage included in this snapshot describes %s.']
    
    def _process_image(self, image):
        image, selected_anchor = self.resizer(image)
        image_input = self.image_transform(image) # h,w,3 -> 3,h,w
        # rearrange(x,'B C (n1 h) (n2 w) -> (B n1 n2) C h w', n1=self.down_sample[0], n2=self.down_sample[1])
        image_input = rearrange(image_input, 'C (num_h h) (num_w w) -> (num_h num_w) C h w', h=self.image_size[0], w=self.image_size[1])

        anchor = self.anchors[selected_anchor] # w,h
        patch_position = torch.cat([
            repeat(torch.arange(anchor[0]), 'num_h -> num_h num_w 1', num_w=anchor[1]),
            repeat(torch.arange(anchor[1]), 'num_w -> num_h num_w 1', num_h=anchor[0])],dim=2)
        patch_position = rearrange(patch_position, 'num_h num_w p-> (num_h num_w) p', p=2) # num_patch, (ph,pw)
        return image_input, patch_position

    def _process_text(self, text):
        if isinstance(text["prompt"], list):
            prompt = random.choice(text["prompt"])
        else:
            prompt = text["prompt"]

        # 分离<image>和文本
        image_token_str = text["text"][:text["text"].rfind('<image>')+len('<image>')]
        area_text =  text["text"][text["text"].rfind('<image>')+len('<image>'):]
        text["text"] = '\''+ text["text"] +'\''
        ocr_instruct=random.choice(self.ocr_instructions)
        text_input = dict(
            prompt=text["prompt"],
            completion=image_token_str + ocr_instruct % area_text,
        )
        return text_input

    def __call__(self, image, text):
        assert image or text
        patch_position = None
        if image:
            image_input, patch_position = self._process_image(image)
        else:
            image_input = None

        if text:
            text_input = self._process_text(text)
        else:
            text_input = None
        return image_input, text_input, patch_position

@PROCESSORS.register_module()
class DocSFTProcessor(DocPretrainProcessor):
   
    def _process_text(self, text):
        if isinstance(text["prompt"], list):
            prompt = random.choice(text["prompt"])
        else:
            prompt = text["prompt"]

        text_input = dict(
            prompt=text["prompt"],
            completion=text["text"],
        )
        return text_input


@PROCESSORS.register_module()
class DocNoCutProcessor:
    def __init__(self, image_size=224, anchors=None):
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.text_transform = None

    def __call__(self, image, text):
        assert image or text
        patch_position = None
        if image:
            image_input = self.image_transform(image).unsqueeze(0)
            patch_position = torch.zeros(1,2).long()
        else:
            image_input = None

        if text:
            if isinstance(text["prompt"], list):
                prompt = random.choice(text["prompt"])
            else:
                prompt = text["prompt"]
            text_input = dict(
                prompt=prompt,
                completion=text["text"],
            )
        else:
            text_input = None
        return image_input, text_input, patch_position

@PROCESSORS.register_module()
class DocNewSFTProcessor(DocSFTProcessor):
    '''
    新Processor用make_text预组织文本 下游task如果需要定制 可以继承这个类进行改进
    {
        "image": ["xxx"],
        "prompt": "", 
        "text": "", 
        "system_instruction": "", 
        "conversations": [
            {"from": "user", "value": "<image>"}, 
            {"from": "user", "value": "Which country has longest bar?"}, 
            {"from": "assistant", "value": "Nigeria"}
        ], 
        "task_type": "qa_sft"
    }
    '''

    def build_text(self, data):
        fin_text = ''
        if 'system_instruction' in data:
            if isinstance(data['system_instruction'], str):
                fin_text+=data['system_instruction']
            elif isinstance(data['system_instruction'], list):
                fin_text+=random.choice(data['system_instruction'])
            else:
                pass
            if not fin_text.endswith('\n'):
                fin_text += '\n'
        
        for cv in data['conversations']:
            if cv['from'] == 'user':
                fin_text+='Human: '+cv['value']
            elif cv['from'] == 'assistant':
                fin_text+='AI: '+cv['value']
            if not fin_text.endswith('\n'):
                fin_text += '\n'
        return fin_text


@PROCESSORS.register_module()
class DocNewMultiScaleSFTProcessor(DocNewSFTProcessor):
    def _process_image(self, image):
        nocut_image = self.image_transform(self.old_resizer(image)).unsqueeze(0)

        image, selected_anchor = self.resizer(image)
        image_input = self.image_transform(image) # h,w,3 -> 3,h,w
        # rearrange(x,'B C (n1 h) (n2 w) -> (B n1 n2) C h w', n1=self.down_sample[0], n2=self.down_sample[1])
        image_input = rearrange(image_input, 'C (num_h h) (num_w w) -> (num_h num_w) C h w', h=self.image_size[0], w=self.image_size[1])

        anchor = self.anchors[selected_anchor] # w,h
        patch_position = torch.cat([
            repeat(torch.arange(anchor[0]), 'num_h -> num_h num_w 1', num_w=anchor[1]),
            repeat(torch.arange(anchor[1]), 'num_w -> num_h num_w 1', num_h=anchor[0])],dim=2)
        patch_position = rearrange(patch_position, 'num_h num_w p-> (num_h num_w) p', p=2) # num_patch, (ph,pw)
        
        image_input = torch.cat([nocut_image, image_input], dim=0)
        patch_position = torch.cat([torch.ones(1,2).long()*self.anchor_max, patch_position], dim=0) # 切片id为0~8
        return image_input, patch_position

if __name__ == '__main__':
    pre_pc = DocPretrainProcessor()
    pass