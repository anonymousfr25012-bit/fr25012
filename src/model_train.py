import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
import argparse

from data_loader import BoomerBoltsDataset  # Import your custom dataset class
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from model import get_model

model_file = 'models/new_bolts/custom_faster_rcnn_chkpt_49.pth'
model_file = "models/new_real_mine_more_epoch/custom_faster_rcnn_chkpt_99.pth"

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


def train(model, args):
    # Define your custom dataset and data loaders
    print("training in depth only model:", args.depth_only)
    dataset = BoomerBoltsDataset(root='/boomer_localization/boomer2_tools/data/GarpenbergScans2024-11-07/data/', transforms=None, depth_only=args.depth_only) #get_transform(train=True))  # Modify as needed
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # # Number of classes (including background)
    # num_classes = 2
    #
    # # Initialize the backbone network (e.g., resnet18)
    # backbone = torchvision.models.resnet18(weights=None)
    # backbone = backbone.float() #make sure network is in float precision
    # backbone.out_channels = 512
    # in_features = backbone.fc.in_features
    #
    # # Define an anchor generator
    # rpn_anchor_generator = AnchorGenerator(
    #     sizes=((32, 64, 128, 256),),
    #     aspect_ratios=((0.5, 1.0, 2.0),)
    # )
    #
    # # Define the ROI feature aligner
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    #     featmap_names=['0'], output_size=7, sampling_ratio=2
    # )
    #
    # # Create the Faster R-CNN model
    # model = FasterRCNN(
    #     backbone,
    #     num_classes,
    #     rpn_anchor_generator=rpn_anchor_generator,
    #     box_roi_pool=roi_pooler
    # )
    #
    # # Set the model to training mode
    # model.train()

    # Define your loss function (e.g., using torchvision's Faster R-CNN loss)
    #criterion = torchvision.models.detection.loss.FasterRCNNLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

   # Load pre-trained parameters
    model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))

    model.to(device)
    model.train()

    # Define your optimizer (e.g., SGD)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)

    num_epochs=100
    print_frequency=10
    save_frequency=500
    # Lists to store training loss and iteration number for visualization
    train_losses = []
    iterations = []

    # Training loop
    for epoch in range(num_epochs):
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in targets]

    #        targets = [{k: v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            iterations.append(i + epoch * len(data_loader))

            if i % print_frequency == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Iteration [{i}/{len(data_loader)}] Loss: {loss.item()}")
            # if i % save_frequency == 0:
                # torch.save(model.state_dict(), 'models/custom_faster_rcnn_chkpt_{}_{}.pth'.format(epoch,i))
        torch.save(model.state_dict(), 'models/custom_faster_rcnn_chkpt_{}.pth'.format(epoch))    

    # Save the trained model
    # torch.save(model.state_dict(), 'models/custom_faster_rcnn.pth')

    # Plot the training curve
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, train_losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Faster R-CNN with Custom Backbone')
    parser.add_argument('--backbone', choices=['resnet18', 'resnet50'], default='resnet18',
                        help='Backbone network for Faster R-CNN (default: resnet18)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes in the dataset (default: 2 -- 1 class (wheat) + background)')
    parser.add_argument('--depth_only', action='store_true', help='Process only depth images.')

    args = parser.parse_args()
    # Get the model with the specified backbone and number of classes
    model = get_model(args.backbone, args.num_classes)

    train(model, args)
