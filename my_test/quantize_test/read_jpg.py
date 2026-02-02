# 导入OpenCV库
import cv2

# ********** 请修改为你的JPG图片路径 **********
image_path = "bear_21.jpg"
# 读取JPG图片，cv2.imread默认以BGR格式读取并存储为numpy数组
# 第二个参数cv2.IMREAD_COLOR表示读取彩色图像（忽略透明通道，JPG无透明通道）
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 异常处理：判断图片是否读取成功（路径错误/文件损坏会导致image为None）
if image is None:
    raise FileNotFoundError(f"无法读取图片，请检查路径是否正确：{image_path}")

# 获取图片左上角第一个像素点的BGR分量（OpenCV图像数组索引为[行, 列, 通道]，第一个像素是第0行第0列）
first_pixel_bgr = image[0, 3] # (62, 137, 81)
# first_pixel_bgr = image[577, 49] #(59, 97, 121)
# 解析B、G、R三个分量（数组顺序为B→G→R，直接解包即可）
b_component, g_component, r_component = first_pixel_bgr

# 打印第一个像素点的BGR分量，清晰展示各通道数值
print(f"第一个像素点的BGR分量：")
print(f"蓝色(B)分量：{b_component}, {b_component/255}")
print(f"绿色(G)分量：{g_component}, {g_component/255}")
print(f"红色(R)分量：{r_component}, {r_component/255}")
print(f"BGR完整值：{first_pixel_bgr}")