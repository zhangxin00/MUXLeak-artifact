#include "readdata.hpp"
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/time.h>

#define Xil_In32(Addr)  (*(volatile u32 *)(Addr))
#define XTDC_ReadReg(addr, offset) \
    Xil_In32((addr) + (offset))
#define XTDC_DATA_OFFSET 0x00
#define XTDC_ReadData(BaseAddr) \
    XTDC_ReadReg((BaseAddr), XTDC_DATA_OFFSET)
#define CORE_ATTACKER 1

typedef struct
{
    u16 DeviceId;
    u32 BaseAddr;
    u8 Depth;
    u8 Count;
} XTDC_Config;

typedef struct
{
    XTDC_Config Config;
    u32 IsReady;
    u32 IsStarted;
    u32 Fine;
    u32 Coarse;
} XTDC;

int XTDC_CfgInitialize(XTDC *InstancePtr, XTDC_Config *ConfigPtr)
{

    if (InstancePtr->IsStarted == 1)
    {
        return 1;
    }
    InstancePtr->Config.DeviceId = ConfigPtr->DeviceId;
    InstancePtr->Config.BaseAddr = ConfigPtr->BaseAddr;
    InstancePtr->Config.Depth = ConfigPtr->Depth;
    InstancePtr->Config.Count = ConfigPtr->Count;
    InstancePtr->IsStarted = 0;
    InstancePtr->IsReady = 1;
    InstancePtr->Coarse = 0;
    InstancePtr->Fine = 0;

    return 0;
}

void pin_to_core(int core)
{
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);

  pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

void* thread_fun(void* dummy)
{
    pin_to_core(CORE_ATTACKER);
    XTDC tdc_inst;
    XTDC_Config XTDC_ConfigTable;
    XTDC_ConfigTable.DeviceId = 1;
    // XTDC_ConfigTable.BaseAddr=0x43C00000;
    int init;
    FILE *fp; 
    fp = fopen("trace.txt", "w");
    if (fp == NULL)                    //判断文件是否成功打开
    {
        perror("File open failed!\n");
       //  return EXIT_FAILURE;
    }

    int HW;
    if ((HW = open("/dev/mem", O_RDWR)) < 0) // open Zynq memory
    {
        perror("open failed\n");
       // return EXIT_FAILURE;
    }

    // channel 0
    XTDC_ConfigTable.BaseAddr =(u32)mmap(NULL, 16 * sysconf(_SC_PAGESIZE), PROT_READ | PROT_WRITE, MAP_SHARED, HW, 0x43C10000); // 64K
    init = XTDC_CfgInitialize(&tdc_inst, &XTDC_ConfigTable);
    XTDC_ConfigTable.Depth = 8;
    XTDC_ConfigTable.Count = 4;
    // timeval nowtime;
    // gettimeofday(&nowtime, NULL);
    // fprintf(fp,"%ld\n",(nowtime.tv_sec%100)*1000000+nowtime.tv_usec);
    while (1)
    {

        // printf("data: %ld\n", XTDC_ReadData(tdc_inst.Config.BaseAddr);
        // gettimeofday(&nowtime, NULL);
        fprintf(fp,"%ld\n", XTDC_ReadData(tdc_inst.Config.BaseAddr));
    }
}
