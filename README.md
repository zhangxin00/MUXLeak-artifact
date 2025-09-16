# MUXLeak

This repository contains the experiments of evaluation and case studies discussed in the paper :
* "MUXLeak: Exploiting Multiplexers as A Power Side Channel against Multi-tenant FPGAs" (TCAD 2025).
  

MUXLeak is a novel on-chip sensor that exploits Multiplexer (MUX) to craft a stealthy power side channel. We mount different power side channel attacks via MUXLeak successfully.

## Tested Setup

### Software dependencies

In order to run the experiments and proof-of-concepts, the following prerequisites need to be fulfilled:

* Linux installation
  * Build tools (gcc, make)
  * Python 3
  * Vivado

### Hardware dependencies

We evaluate MUXLeak on multiple machines across various FPGA architectures summarized as follows, including two System-on-Chip (SoC) boards and one custom-built machine. To build the custom-built machine, the Basys 3 board is connected to a Huawei MateBook 14 laptop via UART and powered through the USB interface.

| Property          | PYNQ-Z2    | ALINX AXU3EGB          | Basys 3    |
| ----------------- | ---------- | ---------------------- | ---------- |
| FPGA Family       | Zynq-7000  | Zynq UltraScale+ MPSoC | Artix-7    |
| Number of MUXes   | 39,900     | 61,740                 | 15,600     |
| Logic Cells       | 85,000     | 154,350                | 33,280     |
| FPGA Core Voltage | 0.95∼1.05V | 0.825∼0.876V           | 0.95∼1.05V |
| CPU Model         | Cortex-A9  | Cortex-A53             | i7-10510U  |
| CPU Cores         | 2          | 4                      | 4          |
| Memory (DRAM)     | 512 MB     | 5 GB                   | 16 GB      |


## Materials

This repository contains the following materials:

* `E1-Extracting_AES_Keys`: This project contains a complete hardware design (including MUXLeak and AES) along with software code for data interaction with an FPGA via serial communication.
* `E2-Stealing_DNN_Model`: This project includes a complete hardware design (featuring MUXLeak and neural network operators), software code for sampling the output of MUXLeak during network inference, and code for analyzing sampled data to infer the network model structure.
* `E3-Building_Covert_Channels`: This project features a hardware design that incorporates MUXLeak, along with code for sending and receiving data through covert channels, as well as code for calculating the bit error rate.

## How should I cite this work?

Please use the following BibTeX entry:

```latex
@inproceedings{Zhang2025MUXLeak,
  year={2025},
  title={MUXLeak: Exploiting Multiplexers as A Power Side Channel against Multi-tenant FPGAs},
  booktitle={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  author={Xin Zhang, Jiajun Zou, Zhi Zhang, Qingni Shen, Yansong Gao, Jinhua Cui, Yusi Feng, Zhonghai Wu, Derek Abbott}
}
