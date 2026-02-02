import shutil
import os

src_file = "preTreatConfig_debug.json.back"
dst_file = "quantization_config.json"


def run_ptq():
    os.system(f"../../build/quantized.out yolov5s_v7.mnn yolov5s_v7_offline_quant.mnn {dst_file}")


def check_sensitive_layer():
    for i in range(23, 231):
        shutil.copy2(src_file, dst_file)
        with open(dst_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < i:
            print(f"the line number of {dst_file} is {len(lines)} < {i}")
            break
        print(f"\n================================= check: {lines[i-1]}")
        del lines[i-1]

        with open(dst_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        run_ptq()


if __name__ == "__main__":
    check_sensitive_layer()
