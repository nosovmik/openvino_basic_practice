# The OpenVINO toolkit practice - basic classification - step 2

This page describes further improvements that can be done after completing Step 1

## Code

### Check FPS

First let's integrate some measurements of frames per second
```
#include <chrono>
```
Do inference in cycle
```
    int count = 0;
    using namespace std::chrono;
    auto start_time = steady_clock::now();
    auto end_time = steady_clock::now() + seconds(10);
    while (steady_clock::now() < end_time) {
        count++;
    }
    end_time = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end_time - start_time).count();
    auto fps = 1000.f * count / diff;
    std::cout << count << " frames completed within " << diff << "ms. FPS=" << fps << "\n";
```

After this you'll be able to see how many images can be processed per second

### Remove naive code

Remove and remove the following blocks of code
```
    // Get pointer to allocated input buffer (already float32)
    ...
```

and 

```
// Apply mean/scales
```

Instead we can create Preprocessing steps right after loading of `cnnNetwork`

```
    // Modify inputs format before ‘LoadNetwork’
    float means[3] = {103.94f, 116.78f, 123.68f};
    float scales[3] = {58.8235f, 58.8235f, 58.8235f};
    auto input_info = cnnNetwork.getInputsInfo()["data"];
    input_info->setLayout(InferenceEngine::Layout::NHWC);
    input_info->setPrecision(InferenceEngine::Precision::U8);
    input_info->getPreProcess().init(3);
    input_info->getPreProcess().setVariant(InferenceEngine::MEAN_VALUE);
    for (int c = 0; c < 3; c++) {
        input_info->getPreProcess()[c]->meanValue = means[c];
        input_info->getPreProcess()[c]->stdScale = scales[c];
    }

```

After this, loaded network `execNetwork` will expect input just as our `image`

```
    // After ‘CreateInferRequest’ and load image
    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, 3, size, size},
                                      InferenceEngine::Layout::NHWC);
    auto blob = InferenceEngine::make_shared_blob<uint8_t>(tDesc, image.ptr());
    inferRequest.SetBlob("data", blob);
```

## Model Optimizer

### Integrate mean/scales into execution graph

Preprocessing steps written above will be always executed on CPU which may not be always desired, at least if we want to use GPU, we want to utilize it as much as possible.
Model Optimizer tool has an ability to inject mean/scale preprocessing steps into execution graph. In our case we can do this with the following command

```
c:\openvino_basic_practice>python "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --input_model public\mobilenet-v2\mobilenet-v2.caffemodel --mean_values [103.94,116.78,123.68] --scale 58.8235 --model_name m2
```

This will create `m2.xml` and `m2.bin` files. XML will now contain Multiply+Add operations before first convolution

### Alternative: use 'converter.py' tool from OMZ

Use 'converter.py' tool to convert model from OMZ to IR suitable for OMZ demos

```
c:\openvino_basic_practice>python "%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader\converter.py" --name mobilenet-v2
```

You could see output like

```
[ SUCCESS ] Generated IR version 10 model.
[ SUCCESS ] XML file: c:\openvino_basic_practice\public\mobilenet-v2\FP32\mobilenet-v2.xml
[ SUCCESS ] BIN file: c:\openvino_basic_practice\public\mobilenet-v2\FP32\mobilenet-v2.bin
[ SUCCESS ] Total execution time: 18.11 seconds.
```

After this step, let's assume that you have `c:\openvino_basic_practice\m2.xml` model with intergated mean/scales there

## Back to code

Change path to IR file:
```
    std::cout << "Reading...";
    // Read model from IR file (XML/BIN)
    InferenceEngine::CNNNetwork cnnNetwork = engine.ReadNetwork("c:\\openvino_basic_practice\\m2.xml");
    std::cout << "Done\n";
```

Comment out all code related to mean/scale preprocessing

```
//    float means[3] = {103.94f, 116.78f, 123.68f};
//    float scales[3] = {58.8235f, 58.8235f, 58.8235f};
//    input_info->getPreProcess().init(3);
//    input_info->getPreProcess().setVariant(InferenceEngine::MEAN_VALUE);
//    for (int c = 0; c < 3; c++) {
//        input_info->getPreProcess()[c]->meanValue = means[c];
//        input_info->getPreProcess()[c]->stdScale = scales[c];
//    }

```

Done. Ensure that inference results now are the same as on step 1
