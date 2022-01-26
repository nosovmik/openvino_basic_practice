# The OpenVINO toolkit practice - basic classification

## Preparation

- Install Python 3.6/3.7/3.8 64-bit - it is needed for model downloading and converting to Intermediate Representation file (IR)
- Install Visual Studio 2015/2017
- Install latest CMake, either via following URL: https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2-win64-x64.msi or via official website cmake.org
- Install the OpenVINO toolkit:
  - Download URL: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit-download.html
  - Distribution: online and offline
  - Version: 2021.4.2 LTS (This is the latest available version which is used for this sample)
  - Type: offline
  - During installation - keep all checkboxes on (Inference Engine, Model Optimizer, Open Model Zoo, OpenCV)
- Clone this repository

## CMakelists.txt

Setting minimum requirements:
```
cmake_minimum_required (VERSION 3.10)
```

Name our project:
```
project(ov_practice)
```

Lets find necessary packages OpenCV and Inference Engine
```
find_package(OpenCV REQUIRED)
find_package(InferenceEngine 2021.4.2 REQUIRED)
```

Create our executable file:
```
add_executable(ov_practice ov_practice1.cpp)
```

Link additional libraries, in our case these are OpenCV and Inference Engine:
```
target_link_libraries(ov_practice ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})
```

## How to build project

We will build project using CMake.

For this, create separate directory for build files, e.g. 'build'
```
c:\openvino_basic_practice>mkdir build
c:\openvino_basic_practice>cd build
c:\openvino_basic_practice\build>
```


To allow CMake find OpenVINO, we need to run 'setupvars.bat' script in command prompt for Visual Studio

```
c:\openvino_basic_practice\build>c:\Program Files (x86)\Intel\openvino_2021\bin\setupvars.bat
```

If you don't get any warnings, you can create .sln file for Visual Studio

```
c:\openvino_basic_practice\build>cmake c:\openvino_basic_practice
```

Then you can open generated ov_practice.sln in Visual Studio

If you have Visual Studio, you can build project from there now. If you don't you can build it from command line:
```
c:\openvino_basic_practice\build>cmake --build . --config Release
```


## How to download pre-trained classification model

You can see list of available models in OMZ here: https://docs.openvino.ai/latest/omz_models_group_public.html

Or check your OpenVINO installation folder (C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\open_model_zoo\models)

In same terminal (where setupvars.bat was executed) run the following command:

```
c:\openvino_basic_practice>python "%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader\downloader.py" --name mobilenet-v2
```

Once original model is downloaded, convert it to IR
```
c:\openvino_basic_practice>python "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --input_model public\mobilenet-v2\mobilenet-v2.caffemodel
```

After this you'll have the following files in c:\openvino_basic_practice: mobilenet-v2.xml, mobilenet-v2.bin, mobilenet-v2.mapping


## Code (step 1)

### What we use

We need to include some modules, like standard modules:
```
#include <iostream>
```
then OpenCV and Inference Engine:
```
#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>
```

### Create engine

First create InferenceEngine::Core object and execute some method

```
int main(int argc, char *argv[])
{
    InferenceEngine::Core engine;
    auto devices = engine.GetAvailableDevices();
    std::cout << "Available devices :";
    for (const auto& device: devices) {
         std::cout << " " << device;
    }
    std::cout << std::endl;
}

```

This will print out list of available devices, like "CPU GPU GNA"

### Prepare model

Read model from IR file:
```
    std::cout << "Reading...";
    // Read model from IR file (XML/BIN)
    InferenceEngine::CNNNetwork cnnNetwork = engine.ReadNetwork(
        "c:\\openvino_basic_practice\\mobilenet-v2.xml",
        "c:\\openvino_basic_practice\\mobilenet-v2.bin");
    std::cout << "Done\n";
```

Load model to CPU

```
    std::cout << "Loading...";
    InferenceEngine::ExecutableNetwork execNetwork = engine.LoadNetwork(cnnNetwork, "CPU");
    std::cout << "Done\n";
```

Create Infer Request

```
    std::cout << "Creating infer request...";
    InferenceEngine::InferRequest inferRequest = execNetwork.CreateInferRequest();
    std::cout << "Done\n";
```

Simple steps are done, now it's time to read some image and pass it to loaded model

### Read image using OpenCV

```
    std::string img_path = "c:\\openvino_basic_practice\\car.png";
    size_t size = 224;
    cv::Mat image = cv::imread(img_path);
    if (!image.ptr()) { std::cout << "Image load failed\n"; return -1; }
    cv::resize(image, image, cv::Size(size, size));
    std::cout << image.channels() << " " << image.rows << " " << image.cols << "\n";
```

Last line shall print that image now has 3 channels, 224 rows and 224 columns. Actual data buffer has 224x224x3 dimension (a.k.a HWC, H-first height, W-then width, C-channels)

Exercise: find API to get this '224' from loaded model (`cnnNetwork`)

### Convert it to model's expectation

We have `image` object which has 224x224x3 dimensions, and 'unsigned 8-bit' data type (B, G, R - from 0 to 255).

But model (mobilenet-v2, https://docs.openvino.ai/latest/omz_models_model_mobilenet_v2.html) expects different data: float, 1x3x224x224, BGR order

Naive manual conversion can look like
```
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
    float means[3] = {103.94, 116.78, 123.68};
    float scales[3] = {58.8235, 58.8235, 58.8235};
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < size; h++) {
            for (int w = 0; w < size; w++) {
                int dst_index = c * size * size + h * size + w;
                blobData[dst_index] = (blobData[dst_index] - means[c]) / scales[c];
            }
        }
    }
```

### Start inference

Start inference and get result

```
    std::cout << "Running inference...\n";
    inferRequest.Infer();
    std::cout << "Done\n";


    // Get results array (1x1000)
    auto* result = inferRequest.GetBlob("prob")->as<InferenceEngine::MemoryBlob>();
    float* resultData = result->rmap().as<float*>();
```

Exercise: find out API to get name of output from model. This is always good to avoid hard-coding like "prob"

### Look through results and find out class with biggest probability

```
    int bestId = -1;
    float bestVal = 0;
    for (int i = 0; i < 1000; i++) {
        if (bestVal < resultData[i]) {
            bestVal = resultData[i];
            bestId = i + 1;
        }
    }
    std::cout << "BEST: class = " << bestId << ", prob = " << bestVal << std::endl;
```

You can check the actual class of object at C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\demo\squeezenet1.1.labels


## Step 2 - check performance and improve
Since we want to see fps, we need know time of each frame process:
```

        auto startTime = std::chrono::steady_clock::now();
```
Now we ready ro read first frame from camera:
```
        cv::Mat frame;
        cap.read(frame);
```
Then prepare it for Inference Engine request (the function `wrapMat2Blob` is defined in <utils/ocv_common.hpp> and perform necessary transformation from cv::Mat to IE supported format):
```
        InferenceEngine::Blob::Ptr imgBlob = wrapMat2Blob(frame);
        inferRequest.SetBlob(inputName, imgBlob);
```
After setting data we can launch our model and get result:
```
        inferRequest.Infer();
        InferenceEngine::Blob::Ptr result = inferRequest.GetBlob(outputName);
```

Then model gives us result, it often need some postprocessing (e.g. resize it to frame size). Class `SegmentationModel` already has it and we can use it.
```
        std::unique_ptr<ResultBase> segmentationResult = model->postprocess(inferenceResult);
```    
Now we ready to process our frame with segmentation mask and perform necessary transformation. The code below has few different transformation types:
```
        cv::Mat outFrame;
        switch (type)
        {
        case DELETE:
            outFrame = remove_background(frame, segmentationResult->asRef<ImageResult>());
            break;
        case BACKGROUND:
            outFrame = remove_background(frame, background, segmentationResult->asRef<ImageResult>());
            break;
        case BLUR:
            outFrame = blur_background(frame, segmentationResult->asRef<ImageResult>());
            break;
        default:
            break;
        }
```
We perform almost all action with this frame and ready to show, but before it we need to update our metrics and add fps value on our frame:
```
        metrics.update(startTime, outFrame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
        cv::imshow("Video", outFrame);
```
Last but not least, lets add some commands, like exit application when `ESC` pressed, or switch transformation type by `TAB` key:
```
        int key = cv::waitKey(1);
        if (key == 27)
            break;
        if (key == 9)
        {
            type++;
            if (type == NONE)
                type = 0;
        }
```
That's all.

### What about image transformations

Above we note few transformation functions, now we define one of them, `replace_background`. This function will find a person on the `frame` using `mask` and replace `background`.
```
cv::Mat remove_background(cv::Mat frame, cv::Mat background, const SegmentationResult& segmentationResult)
```
Because size of frame and background could differ, we should make them equal:
``` 
    cv::resize(background, background, frame.size());
```
The segmentation model we used, has multiple classes, not only person. But we not interested in other classes. So we need to get rid of all other classes except person. We need to know id of our class (`15` for suggested model) which can help perform masking.
```
    const int personLabel = 15;
    cv::Mat personMask = cv::Mat(mask.size(), mask.type(), personLabel);
    cv::compare(mask, personMask, personMask, cv::CMP_EQ);
```
After it we can mask our frame and higlight only person:
```
    cv::Mat maskedFrame;
    cv::bitwise_or(frame, frame, maskedFrame, personMask);
```
Next we perform similar operation with `background` image, but with opposite mask:
```
    cv::Mat backgroundMask;
    cv::bitwise_not(personMask, backgroundMask);
    cv::Mat maskedBackground;
    cv::bitwise_or(background, background, maskedBackground, backgroundMask);
```
Then after concatenation two masked image we get result:
```
    cv::bitwise_or(maskedFrame, maskedBackground, frame);

    return frame;
```
