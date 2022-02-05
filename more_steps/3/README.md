# The OpenVINO toolkit practice - basic classification - step 3

This page describes performance hints after completing Step 2

## Code

### Asynchronous requests

Create several requests
```
    // Create Infer Request
    std::cout << "Creating infer request...";
    int numReq = 32;
    std::vector<InferenceEngine::InferRequest> inferRequests(numReq);
    for (int i = 0; i < numReq; i++) {
        inferRequests[i] = execNetwork.CreateInferRequest();
    }
    std::cout << "Done\n";
```

Set blob with data to each of them

```
    for (int i = 0; i < numReq; i++) {
        auto blob = InferenceEngine::make_shared_blob<uint8_t>(tDesc, image.ptr());
        inferRequests[i].SetBlob("data", blob);
    }
```

And execute them simultaneously in our loop. Note, code is not optimal, we want to restart request #1 as soon as it completes, not when all requests are finished. To handle completion, use InferRequest::SetCompletionCallback

```
    while (steady_clock::now() < end_time) {
        for (int i = 0; i < numReq; i++) {
            inferRequests[i].StartAsync();
        }
        for (int i = 0; i < numReq; i++) {
            inferRequests[i].Wait();
            count++;
        }
    }
```

We'll print results of 1-st inferRequest

```
    // Get results array (1x1000)
    auto* result = inferRequests[0].GetBlob("prob")->as<InferenceEngine::MemoryBlob>();

```

### Set multiple execution streams

For CPU, it is possible to set several config keys, like number of execution threads and number of throughput streams
```
engine.SetConfig({{"CPU_THREADS_NUM", "8"}}, "CPU");
engine.SetConfig({{"CPU_THROUGHPUT_STREAMS", “32"}}, "CPU");

```
