#!/bin/bash

# 遍历 model* 文件夹
for ((i=1; i<=12; i++)); do
  model_dir="./output/model${i}"
  build_model="./build/model${i}"
  if [ -f "$build_model" ]; then

    file_path="./output/model${i}/weights.bin"
    file_size=$(stat -c "%s" "$file_path")
    
    if [ "$file_size" -lt 20000000 ]; then
        echo "Yes"  
    else
        echo "File size is larger than 20MB. Skipping script execution."
        continue
    fi
    
    echo "Processing model${i}"
    # make clean
    rm -f layer.txt trace.txt
    # 复制文件到 firmware 文件夹
    cp -f "$model_dir/weights.bin" /firmware
    cp -f "$model_dir/network.cpp" /firmware
    cp -f "$model_dir/network.hpp" /firmware
    cp -f "./build/model${i}" /firmware

    # 执行 make 操作
    # make
    
    # 循环执行操作 1 次
    for ((j=1; j<=40; j++)); do
      echo "Running make and test ($j/40)"
      rm -f layer.txt trace.txt
      touch layer.txt
      touch trace.txt
      # 运行指令
      ./model${i} FPGA image.bin
      sleep 2s
      # 复制文件到 result 文件夹
      mkdir -p "result/model${i}_${j}"
       
      cp -f layer.txt trace.txt network.cpp "result/model${i}_${j}"
      sleep 2s
    done
    
    rm -f model${i}
    echo "Finished processing model${i}"
  fi
done
