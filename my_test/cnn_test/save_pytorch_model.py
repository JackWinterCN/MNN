from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import requests

model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
