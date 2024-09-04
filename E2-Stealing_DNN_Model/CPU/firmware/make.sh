#!/bin/bash
if [ -d "build" ]; then
	rm -rf build
fi
mkdir build 
# 遍历 model* 文件夹
for ((i=0; i<=2000; i++)); do
  model_dir="output/model${i}"
  if [ -d "$model_dir" ]; then

    file_path="${model_dir}/weights.bin"
    file_size=$(stat -c "%s" "$file_path")
    
    if [ "$file_size" -lt 20000000 ]; then
        echo "Yes"  
    else
        echo "File size is larger than 20MB. Skipping script execution."
        continue
    fi
    
    echo "Processing model${i}"
  #  make clean
    # 复制文件到 firmware 文件夹
    cp -f "$model_dir/weights.bin" ./
    cp -f "$model_dir/network.cpp" ./
    cp -f "$model_dir/network.hpp" ./
    if [ -f "build/model${i}" ]; then
	    continue;
    fi
    make -j9 > /dev/null
    cp -f test build/model${i}
    echo "${model_dir}"
  fi
done
