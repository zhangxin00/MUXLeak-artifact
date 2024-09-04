# Extracting AES Keys

The equipment used in this experiment is the Basys3 development board, which features a 7-series FPGA chip.

## FPGA

In the directory `./FPGA`, there is a complete project archive for the FPGA part of this experiment. Execute the following commands to unzip the project in the current directory and open this Vivado project:

```shell
cd ./FPGA
unzip RMUX4_sensor.zip
cd RMUX4_sensor
vivado RMUX4_sensor.xpr
```

The MUXLeak sensor implementation files are located at: `./RMUX4_sensor.srcs/sources_1/new/mux_sensor.vhd` and `./RMUX4_sensor.srcs/sources_1/imports/rtl/sensor/sensor.vhd`

- In the Vivado project, you can modify the output clock frequency of the `aes_clk` output port in the `clock_generator_inst` IP core to change the operating frequency of the AES. In this project, the default operating frequency of the AES is 20 MHz.
- By modifying the pblock location information in the file `./RMUX4_sensor.srcs/constrs_1/imports/xdc/constraints_MUXLeak.xdc`, you can adjust the layout of MUXLeak on the FPGA. In this project, the default arrangement has MUXLeak and AES located closely together.

- In the Vivado tool, click on the `Generate Bitstream` button to generate the bitstream of the implementation. Program the FPGA with the generated bitstream.


## Software

In the directory `./sw`, there is software code for data interaction with the FPGA via serial communication on the PC. Please ensure you have programmed the FPGA with the bitstream and make sure SW15 on the Basys3 board is in the ON position.

1. Compile the software part in the directory `./sw` by entering the command `make`.

2. Change to the directory `./sw/bin`. The software comes with a help message. Display the help message by typing `./interface -help`. The help message describes all the arguments needed to collect the power traces. For example, to sample the output traces from MUXLeak while using the same key for AES encryption 70,000 times, you can enter the following command: `./interface -k 0 -pt 1 -t 70000 -s`.

   The expected console output during the sampling process is as follows:

```
Trace :69997
Transfering sensor trace...
Sensor trace transfer done!
Key:7d 26 6a ec b1 53 b4 d5 d6 b1 71 a5 81 36 60 5b
Plain text:08 c8 c1 93 2f 03 af b6 f6 12 2f b2 cb d4 81 f0
Cipher text: 1b 35 ca 81 e5 1a ee 75 d7 73 2f f6 db 67 84 10
Trace :69998
Transfering sensor trace...
Sensor trace transfer done!
Key:7d 26 6a ec b1 53 b4 d5 d6 b1 71 a5 81 36 60 5b
Plain text: 1b 35 ca 81 e5 1a ee 75 d7 73 2f f6 db 67 84 10
Cipher text:24 d6 3c co d7 3a 06 db 47 20 88 2d 5a 85 44 9a
```

3. After sampling, extract two files from the `/bin` directory, which contain the MUXLeak output traces and the ciphertexts obtained from each round of encryption (e.g., `sensor_traces_70k.csv` and `ciphertexts.bin`). These files are used to perform Correlation Power Analysis (CPA) and compute the Key Rank Estimation. For the CPA source code, please refer to [RDS/key_rank at main · mirjanastojilovic/RDS (github.com)](https://github.com/mirjanastojilovic/RDS/tree/main/key_rank).

