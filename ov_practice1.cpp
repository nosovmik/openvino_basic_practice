#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

int main(int argc, char *argv[])
{
    InferenceEngine::Core engine;
    auto devices = engine.GetAvailableDevices();
    std::cout << "Available devices :";
    for (const auto& device: devices) {
         std::cout << " " << device;
    }
    std::cout << std::endl;

    std::cout << "Reading...";
    // Read model from IR file (XML/BIN)
    InferenceEngine::CNNNetwork cnnNetwork = engine.ReadNetwork(
        "c:\\openvino_basic_practice\\mobilenet-v2.xml",
        "c:\\openvino_basic_practice\\mobilenet-v2.bin");
    std::cout << "Done\n";
    // Load network to CPU device
    std::cout << "Loading...";
    InferenceEngine::ExecutableNetwork execNetwork = engine.LoadNetwork(cnnNetwork, "CPU");
    std::cout << "Done\n";
    // Create Infer Request
    std::cout << "Creating infer request...";
    InferenceEngine::InferRequest inferRequest = execNetwork.CreateInferRequest();
    std::cout << "Done\n";


    std::string img_path = "c:\\openvino_basic_practice\\car.png";
    int size = 224;
    cv::Mat image = cv::imread(img_path);
    if (!image.ptr()) { std::cout << "Image load failed\n"; return -1; }
    std::cout << image.channels() << " " << image.rows << " " << image.cols << "\n";
    cv::resize(image, image, cv::Size(size, size));

    // Get pointer to allocated input buffer (already float32)
    auto* blob = inferRequest.GetBlob("data")->as<InferenceEngine::MemoryBlob>();
    float* blobData = blob->rwmap().as<float*>();
    // Do conversion manually
    for (int h = 0; h < size; h++) {
       for (int w = 0; w < size; w++) {
           for (int c = 0; c < 3; c++) {
               int src_index = c + w * 3 + h * size * 3;
               int dst_index = c * size * size + h * size + w;
               blobData[dst_index] = image.ptr()[src_index];
            }
        }
    }

    // Apply mean/scales
    float means[3] = {103.94f, 116.78f, 123.68f};
    float scales[3] = {58.8235f, 58.8235f, 58.8235f};
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < size; h++) {
            for (int w = 0; w < size; w++) {
                int dst_index = c * size * size + h * size + w;
                blobData[dst_index] = (blobData[dst_index] - means[c]) / scales[c];
            }
        }
    }

    std::cout << "Running inference...\n";
    inferRequest.Infer();
    std::cout << "Done\n";

    // Get results array (1x1000)
    auto* result = inferRequest.GetBlob("prob")->as<InferenceEngine::MemoryBlob>();
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
