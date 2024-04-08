from einops import rearrange, repeat
import torch
from torchvision import transforms
from PIL import Image, ImageFile
import random
from torchvision.ops.boxes import box_area

from torchvision.transforms.transforms import InterpolationMode
from torchvision.transforms import functional as F
import numpy as np
from icecream import ic

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

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
    # 优先匹配形状接近 再匹配分辨率接近
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

grid_dict = {
    'grid_1':[
        (1,1)],
    'grid_4':[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1)],
    'grid_9':[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1),
        (1,5),(5,1),
        (1,6),(6,1),(2,3),(3,2),
        (1,7),(7,1),
        (4,2),(2,4),(1,8),(8,1),
        (3,3),(1,9),(9,1)],
    'grid_3x3':[
        (3,3)],
    'grid_20':[
        (1, 1), 
        (1, 2), (2, 1), 
        (1, 3), (3, 1), (1, 4), (2, 2), (4, 1), 
        (1, 5), (5, 1), 
        (1, 6), (2, 3), (3, 2), (6, 1), 
        (1, 7), (7, 1), 
        (1, 8), (2, 4), (4, 2), (8, 1), 
        (1, 9), (3, 3), (9, 1), 
        (1, 10), (2, 5), (5, 2), (10, 1), 
        (1, 11), (11, 1), 
        (2, 6), (3, 4), (4, 3), (6, 2), 
        (2, 7), (7, 2), 
        (3, 5), (5, 3), 
        (2, 8), (4, 4), (8, 2), 
        (2, 9), (3, 6), (6, 3), (9, 2), 
        (2, 10), (4, 5), (5, 4), (10, 2)]
}

class DocProcessor():
    def __init__(self, image_size=224, anchors='grid_9', add_global_img=True, add_textual_crop_indicator=False):
        self.add_global_img = add_global_img
        self.add_textual_crop_indicator = add_textual_crop_indicator
        self.media_token= "<|image|>"
        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        # h,w
        anchors = grid_dict[anchors]
        self.anchors = [tuple(_) for _ in anchors]
        self.anchor_max = max([max(_) for _ in self.anchors])
        # xywh -> xyxy
        self.resizer = AnchorResize(image_size=image_size, anchors=anchors, interpolation=InterpolationMode.BICUBIC)
        self.old_resizer = transforms.Resize(image_size,interpolation=InterpolationMode.BICUBIC)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def _process_image(self, images):
        new_images = []
        new_patch_position = []
        num_image_mult = []
        for image in images:
            if self.add_global_img:
                nocut_image = self.image_transform(self.old_resizer(image)).unsqueeze(0)
                
            image, selected_anchor = self.resizer(image)
            image_input = self.image_transform(image) # h,w,3 -> 3,h,w
            # rearrange(x,'B C (n1 h) (n2 w) -> (B n1 n2) C h w', n1=self.down_sample[0], n2=self.down_sample[1])
            image_input = rearrange(image_input, 'C (num_h h) (num_w w) -> (num_h num_w) C h w', h=self.image_size[0], w=self.image_size[1])

            if self.add_global_img:
                image_input = torch.cat([nocut_image, image_input], dim=0)

            anchor = self.anchors[selected_anchor] # w,h
            patch_position = torch.cat([
                repeat(torch.arange(anchor[0]), 'num_h -> num_h num_w 1', num_w=anchor[1]),
                repeat(torch.arange(anchor[1]), 'num_w -> num_h num_w 1', num_h=anchor[0])],dim=2)
            patch_position = rearrange(patch_position, 'num_h num_w p-> (num_h num_w) p', p=2) # num_patch, (ph,pw)

            if self.add_global_img:
                patch_position = torch.cat([torch.ones(1,2).long()*self.anchor_max, patch_position], dim=0)

            new_images.append(image_input)
            new_patch_position.append(patch_position)
            num_image_mult.append(patch_position.shape[0])

        new_images = torch.cat(new_images,dim=0)
        new_patch_position = torch.cat(new_patch_position, dim=0)
        return new_images, new_patch_position, num_image_mult

    def __call__(self, images=None, query=None):
        assert images is not None

        if not isinstance(images, list):
            images = [images]
        image_pils = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            else:
                image = image.convert('RGB')
            # ic(image.size)
            image_pils.append(image)

        image_data, patch_position, num_image_mult = self._process_image(image_pils)

        assert self.media_token in query
        text_list = query.split(self.media_token)
        text = text_list[0]
        image_token_ptr = 0
        for next_text in text_list[1:]:
            if self.add_textual_crop_indicator:
                # generate image placeholders with interleaved texutual crop indicator
                # e.g. <global_img><|image|><crop_img_row0_col0><|image|><crop_img_row0_col1><|image|>...
                for patch_pos in patch_position.tolist():
                    # global non-crop image
                    if patch_pos[0] == self.anchor_max and patch_pos[1] == self.anchor_max:
                        text += '<global_img><|image|>'
                    else:
                        row_col = 'row'+str(patch_pos[0])+'_col'+str(patch_pos[1])
                        text += '<crop_img_'+row_col+'><|image|>'
            else: 
                # generate successive image placeholders for a image, 1 crop img == 1 <|image|>
                text += '<|image|>'*num_image_mult[image_token_ptr]
            text += next_text
            image_token_ptr += 1

        return image_data, patch_position, text