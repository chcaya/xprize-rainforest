from torch import nn
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class Detector(nn.Module):
    def __init__(self,
                 architecture: str,
                 rcnn_backbone_model_resnet_name: str,
                 rcnn_backbone_model_pretrained: bool,
                 box_predictions_per_image: int):
        super(Detector, self).__init__()
        self.architecture = architecture

        if architecture == "basic":
            # Get the Faster R-CNN model
            self.model = get_basic_faster_rcnn(
                rcnn_backbone_model_resnet_name=rcnn_backbone_model_resnet_name,
                rcnn_backbone_model_pretrained=rcnn_backbone_model_pretrained,
                box_detections_per_img=box_predictions_per_image
            )
        else:
            raise ValueError(f"Could not recognize architecture name '{architecture}'")

    def forward(self, images, targets=None):
        return self.model(images, targets)


def get_basic_faster_rcnn(rcnn_backbone_model_resnet_name: str,
                          rcnn_backbone_model_pretrained: bool,
                          box_detections_per_img: int):
    # Conditionally load the pre-trained weights
    if rcnn_backbone_model_pretrained:
        # Assuming ResNet50 backbone for simplicity; adjust as needed for other models
        if rcnn_backbone_model_resnet_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            # weights = ResNet50_Weights.IMAGENET1K_V1
            backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=weights)
        elif rcnn_backbone_model_resnet_name == 'resnet101':
            weights = ResNet101_Weights.DEFAULT
            backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=weights)
        else:
            # Extend this branch for other models as needed
            raise NotImplementedError(f"Pretrained weights for {rcnn_backbone_model_resnet_name} not implemented."
                                      f" But can probably add the logic here.")
    else:
        backbone = resnet_fpn_backbone(backbone_name=rcnn_backbone_model_resnet_name, weights=None)

    # Define an anchor generator
    rpn_anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),  # Adjust the sizes as needed
        aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Adjust the aspect ratios as needed
    )

    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=2,  # 1 class + background
        rpn_anchor_generator=rpn_anchor_generator,
        box_detections_per_img=box_detections_per_img
    )

    return model
