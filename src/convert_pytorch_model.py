import torch
import torchvision

from model import get_model

model_folder = 'models/'
model_folder = 'models/sim_fisheye/'
model_folder = 'models/fine_tuned_from_sim/'
model_folder = 'models/fisheye_depth_only/'
model_folder = 'models/fisheye/'
model_folder = 'models/fine_tuned_from_sim/'
model_folder = 'models/new_bolts/'
model_folder = 'models/new_real_mine_more_epoch/'
model_folder = 'models/fine_tune_1107/'

model_name = 'custom_faster_rcnn_new.pth'
model_name = 'custom_faster_rcnn_chkpt_49.pth'
model_name = 'custom_faster_rcnn_chkpt_99.pth'
model_name = 'custom_faster_rcnn_chkpt_5.pth'
model_name = 'custom_faster_rcnn_chkpt_50.pth'
# model_name = 'custom_faster_rcnn_chkpt_60.pth'

our_model = model_folder + model_name

def convert_tracing():
  ### converting via tracing ###
  # An instance of your model.
  model = torchvision.models.resnet18()

  # An example input you would normally provide to your model's forward() method.
  example = torch.rand(1, 3, 224, 224)

  # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
  traced_script_module = torch.jit.trace(model, example)
  traced_script_module.save("models/traced_resnet_model.pt")
  ##############################

def convert_simple_model():
  ### Test a simple model ###
  class MyModule(torch.nn.Module):
      def __init__(self, N, M):
          super(MyModule, self).__init__()
          self.weight = torch.nn.Parameter(torch.rand(N, M))

      def forward(self, input):
          if input.sum() > 0:
            output = self.weight.mv(input)
          else:
            output = self.weight + input
          return output

  my_module = MyModule(10,20)
  sm = torch.jit.script(my_module)
  sm.save("models/test_module_model.pt")

def convert_fasterrcnn():
  ### Test fasterrcnn ####


  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
  model.eval()

  traced_model = torch.jit.script(model)
  traced_model.save("models/fasterrcnn_resnet50_fpn.pt")

def convert_our_model():
  ### Test our model ###
  # An instance of your model.
  model = get_model("resnet18", num_classes=2)  
  model.load_state_dict(torch.load(our_model,map_location=torch.device('cpu')))
  model.eval()
  # print(model.get_modules())
  # An example input you would normally provide to your model's forward() method.
  example = torch.rand(1, 3, 512, 512)

  # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
  # traced_script_module = torch.jit.trace(model, example)
  sm = torch.jit.script(model)

  # Serializing Your Script Module to a File
  # traced_script_module.save("models/traced_model.pt")
  sm.save(model_folder + model_name[:-3] + "pt")

def convert_our_model_parts():
    model = get_model("resnet18", num_classes=2)
    model.load_state_dict(torch.load('models/custom_faster_rcnn_new.pth',map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('models/custom_faster_rcnn_.pth',map_location=torch.device('cpu')))
    model.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 512, 512)
    print(model.roi_heads)

if __name__ == "__main__":
  #  convert_our_model()
   convert_our_model()