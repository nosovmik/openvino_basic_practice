#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

int main(int argc, char *argv[])
{
    InferenceEngine::Core engine;
    engine.SetConfig({{"CPU_THREADS_NUM", "8"}}, "CPU");
    engine.SetConfig({{"CPU_THROUGHPUT_STREAMS", "32"}}, "CPU");
    auto devices = engine.GetAvailableDevices();
    std::cout << "Available devices :";
    for (const auto& device: devices) {
         std::cout << " " << device;
    }
    std::cout << std::endl;

    std::cout << "Reading...";
    // Read model from IR file (XML/BIN)
    InferenceEngine::CNNNetwork cnnNetwork = engine.ReadNetwork("c:\\openvino_basic_practice\\m2.xml");
    std::cout << "Done\n";

    // Modify inputs format before ‘LoadNetwork’
    auto input_info = cnnNetwork.getInputsInfo()["data"];
    input_info->setLayout(InferenceEngine::Layout::NHWC);
    input_info->setPrecision(InferenceEngine::Precision::U8);

    // Load network to CPU device
    std::cout << "Loading...";
    InferenceEngine::ExecutableNetwork execNetwork = engine.LoadNetwork(cnnNetwork, "CPU");
    std::cout << "Done\n";
    // Create Infer Request
    std::cout << "Creating infer request...";
    int numReq = 32;
    std::vector<InferenceEngine::InferRequest> inferRequests(numReq);
    for (int i = 0; i < numReq; i++) {
        inferRequests[i] = execNetwork.CreateInferRequest();
    }
    std::cout << "Done\n";

    std::string img_path = "c:\\openvino_basic_practice\\car.png";
    size_t size = 224;
    cv::Mat image = cv::imread(img_path);
    if (!image.ptr()) { std::cout << "Image load failed\n"; return -1; }
    cv::resize(image, image, cv::Size(size, size));
    std::cout << image.channels() << " " << image.rows << " " << image.cols << "\n";

    // After ‘CreateInferRequest’ and load image
    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, 3, size, size},
                                      InferenceEngine::Layout::NHWC);

    for (int i = 0; i < numReq; i++) {
        auto blob = InferenceEngine::make_shared_blob<uint8_t>(tDesc, image.ptr());
        inferRequests[i].SetBlob("data", blob);
    }
    int count = 0;
    using namespace std::chrono;
    auto start_time = steady_clock::now();
    auto end_time = steady_clock::now() + seconds(2);
    std::cout << "Running inference...\n";
    while (steady_clock::now() < end_time) {
        for (int i = 0; i < numReq; i++) {
            inferRequests[i].StartAsync();
        }
        for (int i = 0; i < numReq; i++) {
            inferRequests[i].Wait();
            count++;
        }
    }
    end_time = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end_time - start_time).count();
    auto fps = 1000.f * count / diff;
    std::cout << "Done\n";
    std::cout << count << " frames completed within " << diff << "ms. FPS=" << fps << "\n";

    // Get results array (1x1000)
    auto* result = inferRequests[0].GetBlob("prob")->as<InferenceEngine::MemoryBlob>();
    float* resultData = result->rmap().as<float*>();

    int bestId = -1;
    float bestVal = 0;
    for (int i = 0; i < 1000; i++) {
        if (bestVal < resultData[i]) {
            bestVal = resultData[i];
            bestId = i + 1;
        }
        if (resultData[i] > 0.05) {
            std::cout << "class = " << (i+1) << ", prob = " << resultData[i] << std::endl;
        }
    }
    std::cout << "BEST: class = " << bestId << ", prob = " << bestVal << std::endl;

}
