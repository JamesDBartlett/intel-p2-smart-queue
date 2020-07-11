# Smart Queue Monitoring System - App testing on the Intel® DevCloud
### Project as a part of Udacity Intel® Edge AI for IoT Developers Nanodegree Program

## Overview
Writing Python script and job submission script to request Intel **IEI Tank-870** edge node and run inference on the different hardware types (CPU, GPU, VPU, FPGA). 

I also used the **Heterogeneous** plugin which "enables computing for inference on one network on several devices" and **Multi-Device** plugin that "automatically assigns inference requests to available computational devices to execute the requests in parallel"(**OpenVINO Toolkit**).

## Objectives
* Build out the **smart queuing application** to test its performance on the **Intel® DevCloud** using multiple hardware types.
* Submit inference jobs to **Intel® DevCloud** using the `qsub` command.
* Retrieve and review the results.
* Propose hardware device for **retail**, **manufacturing** and **transportation** scenarios.

#### Usage

  required arguments:
  
    --model              The location of the model XML file

  optional arguments:
  
    --device              The device name (CPU, GPU, FPGA, MYRIAD)
                          I used also HETERO:GPU,CPU for gpu_job ,  HETERO:FPGA,CPU for fpga_job, 
                                      MULTI:CPU,GPU for cpu_job,  MULTI:CPU,MYRIAD for vpu_job
    --video               The location of the video file    
    --queue_param         The queue coordinates
    --output_path         The path to the results folder
    --max_people          The max number of people in queue
    --threshold           The probability threshold to select bounding boxes

#### CPU job was submitted to the 
   - Edge Compute Node [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)  with an [Intel® Core™ i5-6500TE processor](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-).
#### GPU job was submitted to the
   - Edge Compute Node with a CPU and GPU [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)  with an [Intel® Core™ i5-6500TE processor](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-). 
   The inference workload was run on the **Intel® HD Graphics 530** integrated GPU.
#### VPU job was submitted to the
   - Edge Compute Node [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)  with an [Intel® Core™ i5-6500TE CPU](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-). 
   The inference workload was run on an **[Intel® Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick)** installed in this  node.
#### FPGA job was submited to the
   - Edge Compute Node [IEI Tank* 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core)  with an [Intel® Core™ i5-6500TE CPU](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-). 
   The inference workload was run on the **[IEI Mustang-F100-A10](https://www.ieiworld.com/mustang-f100/en/)** FPGA card installed in this node.
## Results
**The example**

Comparision of performance the person-detection-retail-0013/FP16/ model across 4 devices for retail scenario. 
The following timings for the model are compared across all 4 devices (CPU, GPU, VPU, FPGA):
- Model Loading Time   [retail scenario bar graph](https://github.com/ireneuszcierpisz/smart-queue-monitoring/blob/master/retail_model-load.png)
- Average Inference Time   [retail scenario bar graph](https://github.com/ireneuszcierpisz/smart-queue-monitoring/blob/master/retail_model-inference-time.png)
- FPS    [retail scenario bar graph](https://github.com/ireneuszcierpisz/smart-queue-monitoring/blob/master/retail_model-FPS.png)

To view the 'Retail CPU' output_video.mp4  [please click here](https://youtu.be/DeEV4EU_HBU).
