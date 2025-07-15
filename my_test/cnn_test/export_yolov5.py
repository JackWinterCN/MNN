import torch
import os

# 设置 GitHub 令牌（可选）
# 如果你想使用令牌，先在 GitHub 上生成一个个人访问令牌，然后将其设置为环境变量
os.environ['GITHUB_TOKEN'] = ''

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# model_trace = torch.jit.trace(model, torch.rand(1, 3, 640, 640))
# model_trace.save('yolov5s_trace.pt')

# model_script = torch.jit.script(model)
# model_script.save('yolov5s_trace.pt')


input_tensor = torch.randn(1, 3, 640, 640)

# 导出模型为 ONNX
torch.onnx.export(
        model,
        input_tensor,
        'yolov5s.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
print("模型已成功导出为 yolov5s.onnx")