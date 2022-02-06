#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

int main(int argc, char *argv[])
{
    InferenceEngine::Core engine;
    engine.SetConfig({{"CPU_THREADS_NUM", "0"}}, "CPU");
    engine.SetConfig({{"CPU_THROUGHPUT_STREAMS", "32"}}, "CPU");
    engine.SetConfig({{"PERF_COUNT", "YES"}}, "CPU");
    int numReq = 32;
    auto devices = engine.GetAvailableDevices();
    std::cout << "Available devices :";
    for (const auto& device: devices) {
         std::cout << " " << device;
    }
    std::cout << std::endl;

    std::cout << "Reading...";
    // Read model from IR file (XML/BIN)
    InferenceEngine::CNNNetwork cnnNetwork = engine.ReadNetwork("c:\\openvino_basic_practice\\b8.xml");
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

    size_t batch = input_info->getTensorDesc().getDims()[0];
    std::cout << "Batch size is " << batch << std::endl;
    // After ‘CreateInferRequest’ and load image
    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {batch, 3, size, size},
                                      InferenceEngine::Layout::NHWC);

    std::vector<std::vector<uint8_t>> batched_buffers(numReq);
    for (int i = 0; i < numReq; i++) {
        batched_buffers[i] = std::vector<uint8_t>(batch * 3 * size * size);
        auto blob = InferenceEngine::make_shared_blob<uint8_t>(tDesc, batched_buffers[i].data());
        for (size_t j = 0; j < batch; j++) {
            memcpy(batched_buffers[i].data() + j * 3 * size * size, image.ptr(), 3 * size * size);
        }
        inferRequests[i].SetBlob("data", blob);
    }
    using namespace std::chrono;
    auto start_time = steady_clock::now();
    auto end_time = steady_clock::now() + seconds(10);
    std::cout << "Running inference...\n";
    int count = 0;
    while (steady_clock::now() < end_time) {
        for (int i = 0; i < numReq; i++) {
            inferRequests[i].StartAsync();
        }
        for (int i = 0; i < numReq; i++) {
            inferRequests[i].Wait();
            count += batch;
        }
    }
    end_time = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end_time - start_time).count();
    auto fps = 1000.f * count / diff;
    std::cout << "Done\n";
    std::cout << count << " frames completed within " << diff << "ms. FPS=" << fps << "\n";

    // Get results array (batch x 1000)
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

    // Display performance counters
    auto perf_map = inferRequests[0].GetPerformanceCounts();
    for (const auto& it : perf_map) {
        if (it.second.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
            std::cout << it.first << ": "; // layer name
            std::cout << std::to_string(it.second.cpu_uSec) << "mcs ";
            std::cout << it.second.exec_type << "\n";
        }
    }
}
