from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_model(rcnn_backbone_model_pretrained: bool,
              box_detections_per_img: int):
    # Load a pre-trained ResNet50 model
    backbone = resnet_fpn_backbone('resnet50', pretrained=rcnn_backbone_model_pretrained)

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
