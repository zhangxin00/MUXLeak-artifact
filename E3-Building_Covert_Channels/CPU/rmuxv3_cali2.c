#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#define XRMUX_DATA_OFFSET 0x00
#define XRMUX_COARSE_OFFSET 0x04
#define XRMUX_FINE_OFFSET 0x08

#define Xil_In32(Addr)  (*(volatile u32 *)(Addr))
#define Xil_Out32(Addr, Value) \
	(*(volatile u32 *)((Addr)) = (Value))


#define XRMUX_ReadReg(addr, offset) \
    Xil_In32((addr) + (offset))
#define XRMUX_WriteReg(addr, offset, data) \
    Xil_Out32((addr) + (offset), (data))
    
typedef uint32_t u32;

int main(){
	int HW;
    if ((HW = open("/dev/mem", O_RDWR)) < 0) // open Zynq memory
    {
        perror("open");
        return EXIT_FAILURE;
    }
    
    long BaseAddr =(long)mmap(NULL, 16 * sysconf(_SC_PAGESIZE), PROT_READ | PROT_WRITE, MAP_SHARED, HW, 0x80002000);
    printf("Calibrating RMUX2-----------------------------------\n");
    int flag = 0;
    
    u32 ini = 1<<31;
    u32 delay = 1<<31;
    u32 delay_fine = 0;
    for(int i=0;i<31;i++){
    	XRMUX_WriteReg(BaseAddr, XRMUX_COARSE_OFFSET, delay);
        for(int j=0;j<24;j++){
           XRMUX_WriteReg(BaseAddr, XRMUX_FINE_OFFSET, delay_fine);
    	   u32 temp1 = XRMUX_ReadReg(BaseAddr, XRMUX_DATA_OFFSET);
	   u32 temp2 = XRMUX_ReadReg(BaseAddr, XRMUX_DATA_OFFSET);
           u32 temp3 = (temp1+temp2)/2;
    	   if(temp3<70 && temp3>60){
    		flag = 1;
                printf("fine need %d\n",j);
    		break;
	   }
           delay_fine = (delay_fine>>1)+ini;
        }
        if(flag){	
           break;
        }
	delay = (delay>>1)+ini;
    }
    if(flag){
    	printf("Calibration done!\n");
        printf("readdata is %d\n", XRMUX_ReadReg(BaseAddr, XRMUX_DATA_OFFSET));
	}else{
		printf("Calibration error!\n");
	}
	return 0;
}
