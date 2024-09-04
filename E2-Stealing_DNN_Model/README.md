# Stealing DNN Model

This experiment utilizes the Pynq-Z2 development board, which is equipped with the Zynq-7000 series SoC chip.

## FPGA

In the directory `./FPGA`, there is a complete project archive for the FPGA part of this experiment. Execute the following commands to unzip the project in the current directory and open this Vivado project:

```shell
cd ./FPGA
unzip RMUXNN.zip
```

In the `./FPGA/ips` directory, there are two IP cores used within the Vivado project:

```
PYNQ_CNN	: A neural network operator that supports operations such as convolutional and pooling layers.
RMUX    	: Implements the AXI interface for MUXLeak and packages it as an IP core.
```

The neural network operator is written in C/C++ and synthesized into an IP core using Vivado HLS. For the C/C++ code of the neural network operator, please refer to [dgschwend/zynqnet: Master Thesis "ZynqNet: An FPGA-Accelerated Embedded Convolutional Neural Network" (github.com)](https://github.com/dgschwend/zynqnet).

To open the hardware project, use the following command:

```shell
cd ./RMUXNN
vivado RMUXNN.xpr
```

1. In the Vivado tool, navigate to `PROJECT MANAGER -- Settings -- IP -- Repository`, and ensure that the hardware project includes the path to the IP cores mentioned above.

2. Then click on the `Generate Bitstream` button to generate the bitstream for the implementation.
3. Finally, click on `File -- Export -- Export Hardware` to export the file with the `.hdf` extension, which will be used by the Petalinux tool to generate the system image.

## convert

The `./convert` directory contains two files: `convert_caffemodel.py` and `convert_jpeg_to_indata_bin.py`, with the following functionalities:

``` 
convert_caffemodel.py         : Reads files with the extensions .prototxt and .caffemodel, converting the neural network model from the Caffe framework into files that can be processed by the neural network operator: network.cpp, network.hpp, and weights.bin.
convert_jpeg_to_indata_bin.py : Converts JPEG format images into binary format files that can be read by the neural network operator.
```

## firmware

The firmware code runs on the ARM core of the PYNQ-Z2, calling the neural network operator on the FPGA to perform inference with different network architectures. The firmware code can also be referenced at [zynqnet/_FIRMWARE at master · dgschwend/zynqnet (github.com)](https://github.com/dgschwend/zynqnet/tree/master/_FIRMWARE). Since the Zynq XC-7Z045 chip is used in the ZynqNet project, certain modifications have been made in this experiment:

 1. Replace the `weights.bin`, `network.cpp`, and `network.hpp` files generated from different network architectures with their corresponding files in the firmware directory.

 2. Given the limited logical resources and insufficient memory on the PYNQ-Z2, the code in `fpga_top.hpp` was modified as follows to reduce computational performance:

    ```c++
    // Number of Processing Elements
    const int N_PE = 1;
    ```

3. Modify the `cpu_top.cpp` file in the original project to execute the function `copy_weights_to_DRAM(net_CPU)` before calculating each layer of the network, which loads the weights of that layer into memory.

## read_data

```
Content: read_data/
  | readdata.hpp           : Declares a function for reading the output of the MUXLeak IP core, named thread_fun. The retrieved output will be written to a file named trace.txt.
  | readdata.cpp           : Implements the functions declared in readdata.hpp.
  | rmuxv3_cali.c          : Calibrates the MUXLeak IP core.
```

Execute the above code on the ARM core of the PYNQ-Z2.

Before using the function in `readdata.hpp` to read MUXLeak output, please calibrate the MUXLeak IP core by executing the command:

```shell
gcc rmuxv3_cali.c -o rmuxv3_cali.out
sudo ./rmuxv3_cali.out
```

Upon successful calibration, the console output should include similar content:

```
Calibration done!
readdata is 67
```

## analysis

Finally, the experiment attempts to steal the DNN model architecture by sampling the MUXLeak output during inference with different networks. To achieve this, the experiment references the open-source code from NASPY and builds a Transformer model. For more details, please refer to [LouXiaoxuan/NASPY: Source code of ICLR 2022 paper "NASPY: Automated Extraction of Automated Machine Learning Models" (github.com)](https://github.com/LouXiaoxuan/NASPY).

Additionally, modifications were made in the open-source code due to the sampling data in our experiment, focusing on the following three files located in the `./analysis` directory:

```
Myparser-blank.py        : Preprocesses the sampled data in trace.txt and generates a dataset in h5 format.
Mytrain_256.py           : Trains a Transformer model with a model input dimension of 256 using the training set from the dataset and saves the model.
Mytest_256.py            : Tests the trained Transformer model on the test set.
```

The experiment compares the attack effects of Transformer models with different input dimensions. To train Transformer models with different dimension values, use the following command:

```python
python Mytrain_256.py --dmodel 256
```
