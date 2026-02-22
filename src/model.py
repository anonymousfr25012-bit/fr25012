import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(backbone_name, num_classes=2):
    # Load the specified backbone network
    print("backbone network:", backbone_name)
    if backbone_name == 'resnet50':
         model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None,
                                                                # rpn_nms_thresh=0.5,
                                                                # box_nms_thresh=0.1,
                                                                # rpn_pre_nms_top_n_train=1000,
                                                                # rpn_pre_nms_top_n_test=500,
                                                                # rpn_post_nms_top_n_train=1000,
                                                                # rpn_post_nms_top_n_test=500,
                                                                )
    else:
        backbone = resnet_fpn_backbone(backbone_name=backbone_name, weights = None, trainable_layers = 5)  #  trainable_layers (int): number of trainable (not frozen) layers starting from final block.  Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.

        # manually assign image mean and std, otherwise they are 3 chanels by default
        # in torchvision/models/resnet.py input chanel is hardcoded in line 197 
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # image_mean = [0.485, 0.485]
        # image_std = [0.229, 0.229]

        # iamge mean and std might need a better value. 
        # image_mean = [0.485, 0.456, 0.406] image_std = [0.229, 0.224, 0.225]
        image_mean = None
        image_std = None
        model = FasterRCNN(backbone, num_classes, image_mean=image_mean, image_std=image_std)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes )

    return model
