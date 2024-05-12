from torch import nn
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.detection import FasterRCNN, RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class Detector(nn.Module):
    def __init__(self,
                 architecture: str,
                 backbone_model_resnet_name: str,
                 backbone_model_pretrained: bool,
                 box_predictions_per_image: int):
        super(Detector, self).__init__()
        self.architecture = architecture

        if architecture == "fasterrcnn":
            # Get the Faster R-CNN model
            self.model = get_basic_faster_rcnn(
                backbone_model_resnet_name=backbone_model_resnet_name,
                backbone_model_pretrained=backbone_model_pretrained,
                box_detections_per_img=box_predictions_per_image
            )
        elif architecture == "retinanet":
            # Get the RetinaNet model
            self.model = get_basic_retinanet(
                backbone_model_resnet_name=backbone_model_resnet_name,
                backbone_model_pretrained=backbone_model_pretrained,
                box_detections_per_img=box_predictions_per_image
            )
        else:
            raise ValueError(f"Could not recognize architecture name '{architecture}'. Supported architectures are: "
                             "['fasterrcnn', 'retinanet'].")

    def forward(self, images, targets=None):
        return self.model(images, targets)


def get_basic_faster_rcnn(backbone_model_resnet_name: str,
                          backbone_model_pretrained: bool,
                          box_detections_per_img: int):

    if backbone_model_pretrained:
        if backbone_model_resnet_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
        elif backbone_model_resnet_name == 'resnet101':
            weights = ResNet101_Weights.DEFAULT
        elif backbone_model_resnet_name == 'resnet152':
            weights = ResNet152_Weights.DEFAULT
        else:
            raise NotImplementedError(f"Pretrained weights for {backbone_model_resnet_name} not implemented."
                                      f" But can probably add the logic here.")
    else:
        weights = None

    backbone = resnet_fpn_backbone(
        backbone_name=backbone_model_resnet_name,
        weights=weights,
        trainable_layers=3              # TODO parametrize this ?
    )

    # Define an anchor generator
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,                             # TODO parametrize this ?
        aspect_ratios=aspect_ratios                     # TODO parametrize this ?
    )

    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=2,  # 1 class + background
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=box_detections_per_img
    )

    return model


def get_basic_retinanet(backbone_model_resnet_name: str,
                        backbone_model_pretrained: bool,
                        box_detections_per_img: int):

    if backbone_model_pretrained:
        if backbone_model_resnet_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
        elif backbone_model_resnet_name == 'resnet101':
            weights = ResNet101_Weights.DEFAULT
        elif backbone_model_resnet_name == 'resnet152':
            weights = ResNet152_Weights.DEFAULT
        else:
            raise NotImplementedError(f"Pretrained weights for {backbone_model_resnet_name} not implemented.")
        backbone = resnet_fpn_backbone(backbone_name=backbone_model_resnet_name, weights=weights, trainable_layers=5)
    else:
        backbone = resnet_fpn_backbone(backbone_name=backbone_model_resnet_name, weights=None, trainable_layers=5)

    # Define an anchor generator
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,                             # TODO parametrize this ?
        aspect_ratios=aspect_ratios                     # TODO parametrize this ?
    )

    # Create the RetinaNet model using the loaded backbone
    model = RetinaNet(
        backbone=backbone,
        num_classes=2,  # 1 class + background
        anchor_generator=anchor_generator,
        detections_per_img=box_detections_per_img,
    )

    return model
