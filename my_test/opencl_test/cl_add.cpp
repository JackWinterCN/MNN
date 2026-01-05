#include <iostream>
#include <vector>
#include <CL/cl2.hpp>  // 对应OpenCL 2.x的C++绑定

// OpenCL内核代码：两数组逐元素相加
const std::string kernel_code = R"(
__kernel void array_add(__global const float *a, 
                       __global const float *b, 
                       __global float *c, 
                       const unsigned int n) {
    // 获取全局索引
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
)";

int main() {
    try {
        // 1. 初始化OpenCL环境
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("找不到OpenCL平台");
        }

        // 选择第一个可用平台
        cl::Platform platform = platforms[0];
        std::cout << "使用平台: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        // 获取该平台上的所有设备
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            std::cerr << "未找到GPU设备，尝试使用CPU设备" << std::endl;
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        }
        if (devices.empty()) {
            throw std::runtime_error("找不到可用的OpenCL设备");
        }

        // 选择第一个可用设备
        cl::Device device = devices[0];
        std::cout << "使用设备: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // 创建上下文
        cl::Context context(device);

        // 创建命令队列
        cl::CommandQueue queue(context, device);

        // 2. 创建程序和内核
        cl::Program program(context, kernel_code);
        cl_int err = program.build({device});
        if (err != CL_SUCCESS) {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            throw std::runtime_error("程序构建失败: " + build_log);
        }

        cl::Kernel kernel(program, "array_add");

        // 3. 创建输入数据
        const unsigned int n = 1024 * 1024;  // 100万个元素
        std::vector<float> a(n), b(n), c(n);

        // 初始化输入数组
        for (unsigned int i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(n - i);
        }

        // 4. 创建OpenCL缓冲区
        cl::Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                           sizeof(float) * n, a.data());
        cl::Buffer buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                           sizeof(float) * n, b.data());
        cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        // 5. 设置内核参数
        kernel.setArg(0, buffer_a);
        kernel.setArg(1, buffer_b);
        kernel.setArg(2, buffer_c);
        kernel.setArg(3, n);

        // 6. 执行内核
        cl::NDRange global(n);  // 全局工作项数量
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
        queue.finish();  // 等待计算完成

        // 7. 读取结果
        queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, sizeof(float) * n, c.data());

        // 8. 验证结果（检查前10个和最后10个元素）
        bool success = true;
        for (int i = 0; i < 10; ++i) {
            if (std::abs(c[i] - (a[i] + b[i])) > 1e-5) {
                success = false;
                std::cerr << "结果验证失败 at " << i << ": " << std::fixed
                          << c[i] << " != " << (a[i] + b[i]) << std::endl;
                break;
            } else {
              std::cout << "结果验证成功 at " << i << ": " << std::fixed
                        << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
            }
        }
        
        if (success) {
            for (int i = n - 10; i < n; ++i) {
                if (std::abs(c[i] - (a[i] + b[i])) > 1e-5) {
                    success = false;
                    std::cerr << "结果验证失败 at " << i << ": " << std::fixed 
                              << c[i] << " != " << (a[i] + b[i]) << std::endl;
                    break;
                } else {
                  std::cout << "结果验证成功 at " << i << ": " << std::fixed
                            << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
                }
            }
        }

        if (success) {
            std::cout << "计算成功! 数组相加结果正确。" << std::endl;
         }

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}