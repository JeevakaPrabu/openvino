// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample_async/main.cpp
* @example classification_sample_async/main.cpp
*/

#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <map>

#include <inference_engine.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>

#include <sys/stat.h>
#include <log/log.h>

using namespace InferenceEngine;

int main(int argc, char *argv[]) {
    try {
        ALOGI("main() Enter");
        // ------------------------------ Parsing and validation of input args ---------------------------------

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        imageNames.push_back("/vendor/etc/openvino/car_resized.bmp");
        if (imageNames.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        ALOGI("Creating Inference Engine");

        // Load inference engine
        std::vector< std::string > pluginPath = { "/vendor/lib64", "/vendor/lib", "/system/lib64", "/system/lib", "" };

        InferencePlugin plugin = PluginDispatcher(pluginPath).getPluginByDevice("CPU");

        ALOGI("Read network files");

        CNNNetReader networkReader;
        /** Read network model **/
        networkReader.ReadNetwork("/vendor/etc/openvino/SqueezeNet_v1.1_modified_fp32.xml");
        networkReader.ReadWeights("/vendor/etc/openvino/SqueezeNet_v1.1_modified_fp32.bin");

        ALOGI("getNetwork files");
        CNNNetwork network = networkReader.getNetwork();

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        ALOGI("Preparing input blobs");

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the device **/
        inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::shared_ptr<unsigned char>> imagesData = {};
        for (const auto & i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(
                    reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
                                    inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        size_t batchSize = network.getBatchSize();
        ALOGI("Batch size is %zu", batchSize);

        // -----------------------------------------------------------------------------------------------------

        OutputsDataMap outputInfo(network.getOutputsInfo());
        // BlobMap outputBlobs;
        std::string firstOutputName;

        for (auto & item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                ALOGI("output data pointer is not valid");
                return 1;
            }

            item.second->setPrecision(Precision::FP32);
        }

        const SizeVector outputDims = outputInfo.begin()->second->getDims();

        bool outputCorrect = false;
        if (outputDims.size() == 2 /* NC */) {
            outputCorrect = true;
        } else if (outputDims.size() == 4 /* NCHW */) {
            /* H = W = 1 */
            if (outputDims[2] == 1 && outputDims[3] == 1) outputCorrect = true;
        }

        if (!outputCorrect) {
            return 1;
        }

        // --------------------------- 4. Loading model to the device ------------------------------------------
        ALOGI("Loading model to the device");
        std::map<std::string, std::string> config;
        config[InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS] = std::to_string(1);
        ExecutableNetwork executable_network = plugin.LoadNetwork(network, config);
        inputInfoItem.second = {};
        outputInfo = {};
        network = {};
        networkReader = {};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        ALOGI("Create infer request");
        InferRequest inferRequest = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        for (auto & item : inputInfo) {
            Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);
            SizeVector dims = inputBlob->getTensorDesc().getDims();
            /** Fill input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = dims[1];
            size_t image_size = dims[3] * dims[2];
            auto data = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }
        }
        inputInfo = {};

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

        double total = 0.0;
        /** Start inference & calc performance **/
        for (size_t iter = 0; iter < 1; ++iter) {
            auto t0 = Time::now();
            inferRequest.Infer();
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
        }


        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        ALOGI("Processing output blobs");
        const Blob::Ptr outputBlob = inferRequest.GetBlob(firstOutputName);
        if (std::fabs(total) < std::numeric_limits<double>::epsilon()) {
            ALOGI("total can't be equal to zero");
            return 1;
        }
        ALOGI("total inference time: %f", total);
        ALOGI("Average running time of one iteration: %f ms", total);
        ALOGI("Throughput: %f FPS", ((1000 * batchSize) / total));

        ALOGI("Classification successful");

        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        ALOGI("Exception in Classification");
        return 1;
    }
    catch (...) {
        ALOGI("Unknown/internal exception happened.");
        return 1;
    }

    ALOGI("Execution successful");
    return 0;
}
