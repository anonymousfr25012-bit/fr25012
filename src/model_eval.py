import numpy as np
import torch
import torchvision
from data_loader import BoomerBoltsDataset
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
from torchvision import utils

#for model
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

from torchvision.ops import box_iou
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import average_precision_score

from model import get_model

from calc4ap.voc import CalcVOCmAP

from tqdm import tqdm

# test_file_folder = './data/test'
# test_file_folder = './data/data_fisheye/test/'

test_file_folder = './data/data_fisheye/test/'
test_file_folder = './data/test_/'
test_file_folder = './data/data_newbolt/'
test_file_folder = './data/data_real_new/'
test_file_folder = './data/GarpenbergScans2024-11-07/data/'

# test_file_folder = './data/data_sim_fisheye_test/'


# model_file = 'models/sim_depth/fisheye.pth'
# model_file = 'models/sim_fisheye/custom_faster_rcnn_chkpt_49.pth'
# model_file = 'models/fisheye/custom_faster_rcnn_chkpt_49.pth'
model_file = 'models/fisheye_depth_only/custom_faster_rcnn_chkpt_49.pth'  
model_file = 'models/fine_tuned_from_sim/custom_faster_rcnn_chkpt_10.pth'
model_file = "models/new_bolts/custom_faster_rcnn_chkpt_49.pth"
model_file = "models/custom_faster_rcnn_chkpt_49.pth"
model_file = "models/new_real_mine_more_epoch/custom_faster_rcnn_chkpt_99.pth"
model_file = "models/fine_tune_1107/custom_faster_rcnn_chkpt_99.pth"

BATCH_SIZE = 1

depth_only = True
depth_only = False
conf_thr = 0.8


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        box1 (list or numpy array): [x_min, y_min, x_max, y_max] of the first box.
        box2 (list or numpy array): [x_min, y_min, x_max, y_max] of the second box.

    Returns:
        float: Intersection over Union (IoU) value.
    """
    # Calculate intersection coordinates
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    # Calculate areas of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def evaluation(all_predictions, all_ground_truths):


    iou_thrs = []
    prec = []
    rets = []
    for iou in np.arange(0.5, 1.0, 0.05):
        voc_ap = CalcVOCmAP(labels=all_ground_truths, preds=all_predictions, iou_thr=iou, conf_thr=0.7)
        ret = voc_ap.get()
        rets.append(ret) 

        tp, fp, fn = ret['total_TP'], ret['total_FP'], ret['total_FN']
        # print(tp, fp, fn)
        p = tp / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(iou, tp, fp, fn, p))
        prec.append(p)
    
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))



def plot(img, prediction, train_labels):

    plt.imshow(img[1,:,:], cmap="jet")
    plt.title("Image "+str(train_labels["image_id"]))
    ax = plt.gca()
    # Show ground truth bounding boxes
    box = train_labels['boxes'][0]
    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=True, color='green', linewidth=2, label='ground truth'))
    for box in train_labels['boxes'][1:]:
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=True, color='green', linewidth=2))
        # plt.text(box[0], box[1], "gt_bolt", color='green', fontsize=12)

    # Filter predictions to show non-overlapping bounding boxes
    count_for_label = 0
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores'].cpu().detach().numpy()):
        box = box.tolist()
        
        if score > conf_thr:
            if count_for_label <1:
                count_for_label += 1
                ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red', linewidth=1, label='detected'))
            else:
                ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red', linewidth=1))
            # plt.text(box[0], box[1], class_labels[label], color='red', fontsize=12)
            plt.text(box[0], box[1], str(np.around(score,2)), color='red', fontsize=12)
    plt.legend()
    plt.show()

class Hook():
    def __init__(self, module):
        # self.hook = module.register_backward_hook(self.hook_fn)
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        print('????')
    def close(self):
        self.hook.remove()


def collate_fn(batch):
    return tuple(zip(*batch))
    
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.25))
        transforms.append(T.RandomVerticalFlip(0.25))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

dataset = BoomerBoltsDataset(test_file_folder, get_transform(train=True),depth_only=depth_only)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn
)

# dataset = CocoDetection(root=test_file_folder, annFile="path_to_annotation_file")
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
# evaluator = CocoEvaluator(coco_gt=dataset.coco, iou_types=["bbox"])

#weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None,
#                                                              rpn_nms_thresh=0.7,
#                                                              box_nms_thresh=0.5,
#                                                              rpn_pre_nms_top_n_train=1000,
#                                                              rpn_pre_nms_top_n_test=500,
#                                                              rpn_post_nms_top_n_train=1000,
#                                                              rpn_post_nms_top_n_test=500,
#                                                              )

### model with resnet18 backbone ###
backbone_name = 'resnet18'
num_classes = 2
model = get_model(backbone_name, num_classes)
model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))

####################################
### model with resnet50 backbone ###
# backbone_name = 'resnet50'
# model = get_model(backbone_name)
# model.load_state_dict(torch.load('models/custom_faster_rcnn.pth',map_location=torch.device('cpu')))
####################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)
#set in eval mode
model.eval()
model.cuda()

hook_box_roi_pool = Hook(model.roi_heads.box_roi_pool)
hook_box_head = Hook(model.roi_heads.box_head)
hook_scores = Hook(model.roi_heads.box_predictor)
hook_rpn = Hook(model.rpn)

all_predictions = []
all_ground_truths = []
all_scores = []
count_p = 0
count_l = 0
with torch.no_grad():
    for _ in tqdm(range(10)):
    # for train_features, train_labels in tqdm(data_loader):
        # Display image and label.
        train_features, train_labels = next(iter(data_loader))
        # if _ < 265:
        #     continue
        batch_size = len(train_features)
        print(f"Feature batch shape: {len(train_features)}")
        print(f"Labels batch shape: {len(train_labels)}")

        img = train_features[0].squeeze()
        # print(img)
        # label = train_labels[0]
        # plt.imshow(img[0,:,:], cmap="jet")
        # plt.show()
        # plt.imshow(img[1,:,:], cmap="jet")
        # plt.show()
        # print(f"Label: {label}")

        
        # Step 2: Initialize the inference transforms
        #preprocess = weights.transforms()
        class_labels = dataset.classes

        # Step 3: Apply inference preprocessing transforms
        # batch = [train_features[0].squeeze()]
        batch = tuple(t.cuda() for t in train_features)
        # Step 4: Use the model and visualize the prediction
        prediction = model(batch)
        print("!!!!!!!!!!!",len(prediction), prediction[0].keys())
        # print(model)
        # print(list(model._modules.items()))
        # print(hook_box_head.output.size())
        # print(hook_scores.input[0].size())
        # print(hook_scores.output[0].size())
        # print(hook_scores.output[1].size())
        
        _, proposals, image_shapes = hook_box_roi_pool.input
        class_logits, box_regression = hook_scores.output
        
        # all_boxes, all_scores, all_labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        # print("num of boxes:", all_boxes[0].size())
        # boxes_features = model.get_modules()
        # print("features:", len(boxes_features), boxes_features[0].size())

        # img = (255*img[1,:,:])
        # img = torch.stack([img,img,img],axis=2)
        # print(img)
        
        # print(prediction['scores'])

        
        # print(all_scores[0])
        # print(prediction[0]['scores'])
        # print((all_scores[0] - prediction['scores']).detach().numpy())
        

        plot(img, prediction[0], train_labels[0])
        # print("!!!!",prediction)
        for p in prediction:
            p_boxes = p['boxes'].cpu().numpy()
            p_scores = p['scores'].cpu().numpy()
            p_labels = p['labels'].cpu().numpy()
            p_image_ids = np.ones_like(p_scores)*count_p
            count_p+=1
        
        for l in train_labels:
            labels_boxes = l['boxes'].cpu().numpy()
            labels = l['labels'].cpu().numpy()
            labels_image_ids = np.ones_like(labels)*count_l
            count_l+=1

        p = [[*row_a, elem_b, elem_c, elem_d] for row_a, elem_b, elem_c, elem_d in zip(p_boxes, p_scores, p_labels, p_image_ids)]
        l = [[*row_a, elem_b, elem_c] for row_a, elem_b, elem_c in zip(labels_boxes, labels, labels_image_ids)]
        
        all_predictions.extend(p)
        all_ground_truths.extend(l)
        # all_scores.extend(prediction[0]['scores'].cpu().numpy())



# Evaluate the model
# print("!!!!!", all_predictions)
evaluation(all_predictions, all_ground_truths)



    # labels = ["bolt" for i in prediction["boxes"]]
    # img = (255*img[0,:,:]).byte()
    # img = torch.stack([img,img,img])
    # box = draw_bounding_boxes(img, boxes=prediction["boxes"],
    #                           labels=labels,
    #                           colors="red",
    #                           width=4, font_size=30)
    # im = to_pil_image(box.detach())
    # im.show()

    #
    # backbone = resnet_fpn_backbone('resnet18', pretrained=False, trainable_layers=5)
    # # Define an anchor generator
    # rpn_anchor_generator = AnchorGenerator(
    #     sizes=((32, 64, 128, 256),),
    #     aspect_ratios=((0.5, 1.0, 2.0),)
    # )
    # # Define the ROI feature aligner
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    #     featmap_names=['0'], output_size=7, sampling_ratio=2
    # )
    # # Create the Faster R-CNN model
    # model = FasterRCNN(
    #     backbone,
    #     2,
    #     rpn_anchor_generator=rpn_anchor_generator,
    #     box_roi_pool=roi_pooler
    # )
    #
    # #set in eval mode
    # model.eval()
    #
    # # Step 2: Initialize the inference transforms
    # preprocess = weights.transforms()
    #
    # # Step 3: Apply inference preprocessing transforms
    # batch0 = [preprocess(train_features[0].squeeze())]
    #
    # batch = [torch.rand(2, 300, 400), torch.rand(2, 500, 400)]
    #
    # # Step 4: Use the model and visualize the prediction
    # prediction = model(batch)[0]
    # labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    # box = draw_bounding_boxes(img, boxes=prediction["boxes"],
    #                           labels=labels,
    #                           colors="red",
    #                           width=4, font_size=30)
    # im = to_pil_image(box.detach())
    # im.show()
