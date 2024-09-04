# Building Covert Channel

The equipment used in this experiment is the ALINX AXU3EGB, which features a Zynq UltraScale+ MPSoC chip.

## FPGA

In the directory `./FPGA`, there is a complete project archive for the FPGA part of this experiment. Execute the following commands to unzip the project in the current directory and open this Vivado project:

```shell
cd ./FPGA
unzip 3eg.zip
cd 3eg
vivado 3eg.xpr
```

- In the `./ip_repo` directory, there is an MUXLeak IP core that implements the AXI interface for the Ultrascale+ architecture FPGA chip. The path to this IP core has been added to the IP Repository in the Vivado project.

- In the Vivado tool, click on the `Generate Bitstream` button to generate the bitstream of the implementation.

  Then, click `File -- Export -- Export Hardware` to export a file with the `.hdf` extension, which will be used by the Petalinux tool to generate the system image.

## CPU

```
Content: CPU/
  |error-rate.c           : Calculates the error rate between sent and received data.
  |receiver.c             : Computes the average output for each bit's sending time by reading outputs from various MUXLeak instances, inferring the sent bit values.
  |rmuxv3_cali.c          : Calibrates a specific MUXLeak IP core.
  |rmuxv3_cali1.c         : Calibrates a specific MUXLeak IP core.
  |rmuxv3_cali2.c         : Calibrates a specific MUXLeak IP core.
  |rmuxv3_cali3.c         : Calibrates a specific MUXLeak IP core.
  |sender.c               : Sends a sequence of bits by stressing the CPU. It agrees on the number of bits to be sent and the sending time for each bit with the receiver.
  |test.sh                : A script used to observe the error rate of this covert channel under different sending times per bit.
```

Execute the above code on the ARM core of the ALINX AXU3EGB.

1. First, calibrate the four MUXLeak IP cores. For example, to use `rmuxv3_cali.c`, execute the command:

   ```shell
   gcc rmuxv3_cali.c -o rmuxv3_cali.out
   sudo ./rmuxv3_cali.out
   ```

   Upon successful calibration, the console output should include similar content:

   ```
   Calibration done!
   readdata is 67
   ```

2. Execute the `test.sh` script to obtain the error rates under different sending times per bit. The results will be written to `result.txt`.

   ```shell
   ./test.sh
   ```

   

