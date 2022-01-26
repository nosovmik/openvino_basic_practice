# The OpenVINO toolkit practice - basic classification - step 4

This page describes further performance hints after completing Step 3

## Model Optimizer

As many probably heard, inference on N images in a batch inputs are usually faster than N separate inference requests

Original model has 1x3x224x224 size. To allow batching we shall change it to 8x3x224x224

Model Optimizer allows to do this with '-b' option

```
c:\openvino_basic_practice>python "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --input_model public\mobilenet-v2\mobilenet-v2.caffemodel --mean_values [103.94,116.78,123.68] --scale 58.8235 -b 8 --model_name m3
```

## Code

### Change input to batched

Get 'batch' number from model

```
    size_t batch = input_info->getTensorDesc().getDims()[0];
```

Filling blobs to infer requests (we have multiple requests created on Step 3)
```
    // After ‘CreateInferRequest’ and load image
    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {batch, 3, size, size},
                                      InferenceEngine::Layout::NHWC);

    for (int i = 0; i < numReq; i++) {
        std::vector<uint8_t> batched_buffer(batch * 3 * size * size);
        auto blob = InferenceEngine::make_shared_blob<uint8_t>(tDesc, batched_buffer.data());
        for (int j = 0; j < batch; j++) {
            memcpy(batched_buffer.data() + j * 3 * size * size, image.ptr(), 3 * size * size);
        }
        inferRequests[i].SetBlob("data", blob);
    }
```

For our metrics, change `count++` to `count += batch`

### Turn on performance counters

Add one more config key to `engine`:
```
    engine.SetConfig({{"PERF_COUNT", "YES"}}, "CPU");
```

After execution, print performance statistics with the following code:

```
    // Display performance counters
    auto perf_map = inferRequests[0].GetPerformanceCounts();
    for (const auto& it : perf_map) {
        if (it.second.status == InferenceEngine::InferenceEngineProfileInfo::EXECUTED) {
            std::cout << it.first << ": "; // layer name
            std::cout << std::to_string(it.second.cpu_uSec) << "mcs ";
            std::cout << it.second.exec_type << "\n";
        }
    }
```
