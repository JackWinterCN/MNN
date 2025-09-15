#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>
using namespace MNN;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./concat_dynamic_shape_test.out model.mnn len" << std::endl;
        return -1;
    }

    const auto poseModel           = argv[1];
    const auto len_str  = argv[2];
    const int len = std::stoi(len_str);

    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(poseModel));
    MNN::ScheduleConfig netConfig;
    netConfig.type      = MNN_FORWARD_CPU;
    netConfig.numThread = 4;
    auto session        = mnnNet->createSession(netConfig);

    auto input0 = mnnNet->getSessionInput(session, "input0");
    auto input1 = mnnNet->getSessionInput(session, "input1");

    if (input0->elementSize() <= 4) {
        mnnNet->resizeTensor(input0, {1, 1, 1, len});
        mnnNet->resizeTensor(input1, {1, 1, 1, len});
        mnnNet->resizeSession(session);
    }
    
    std::shared_ptr<Tensor> inputUser0(new Tensor(input0, Tensor::CAFFE));
    std::shared_ptr<Tensor> inputUser1(new Tensor(input1, Tensor::CAFFE));
    for (int i = 0; i < len; i++) {
        inputUser0->host<float>()[i] = i*i;
        inputUser1->host<float>()[i] = i+i;
    }
    input0->copyFromHostTensor(inputUser0.get());
    input1->copyFromHostTensor(inputUser1.get());

    input0->print();
    input1->print();

    // run...
    {
        AUTOTIME;
        mnnNet->runSession(session);
    }

    // get output
    auto offsets         = mnnNet->getSessionOutput(session, nullptr);
    Tensor offsetsHost(offsets, Tensor::CAFFE);
    offsets->copyToHostTensor(&offsetsHost);

    offsetsHost.print();
    auto shape = offsetsHost.shape();
    for (int i = 0; i < shape.size(); i++) {
        std::cout << "shape[" << i << "] = " << shape[i] << std::endl;
    }

    return 0;
}
