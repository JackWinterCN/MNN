|    model    | hardware | precision | memory_mode | time | ptrace |
| :------: | :------: | :------: | :------: | :------: | :------: |
| xpu_yolov5s | cpu | high | high     | pre: 15.9ms <br> forward: 167.7ms <br> post: 11.7ms | xpu_yolov5_cpu_highP_highM_warmup10_20260414.pftrace |
| qwen2-0.5B | cpu | low | low     | prefill: 229.16 tok/s <br> decode: 46.05 tok/s | qwen2_0.5B_cpu_lowP_lowM_20260414.pftrace |


qwen2_0.5B_cpu_lowP_lowM:
```
prompt tokens num = 17
decode tokens num = 180
vision time = 0.00 s
audio time = 0.00 s
prefill time = 0.07 s
decode time = 3.91 s
sample time = 0.13 s
prefill speed = 229.16 tok/s
decode speed = 46.05 tok/s
```