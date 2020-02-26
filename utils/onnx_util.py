import torch
import torch.onnx
from models.slim import Slim

x = torch.randn(1, 3, 160, 160)
model = Slim()
model.load_state_dict(torch.load("../pretrained_weights/slim_160_latest.pth", map_location="cpu"))
model.eval()
torch.onnx.export(model, x, "../pretrained_weights/slim_160_latest.onnx", input_names=["input1"], output_names=['output1'])
