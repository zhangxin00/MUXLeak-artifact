#include "readdata.hpp"
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/time.h>

#define Xil_In32(Addr)  (*(volatile u32 *)(Addr))
#define XMUX_ReadReg(addr, offset) \
    Xil_In32((addr) + (offset))
#define XMUX_DATA_OFFSET 0x00
#define XMUX_ReadData(BaseAddr) \
    XMUX_ReadReg((BaseAddr), XMUX_DATA_OFFSET)
#define CORE_ATTACKER 1

typedef struct
{
    u16 DeviceId;
    u32 BaseAddr;
    u8 Depth;
    u8 Count;
} XMUX_Config;

typedef struct
{
    XMUX_Config Config;
    u32 IsReady;
    u32 IsStarted;
    u32 Fine;
    u32 Coarse;
} XMUX;

int XMUX_CfgInitialize(XMUX *InstancePtr, XMUX_Config *ConfigPtr)
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
    XMUX mux_inst;
    XMUX_Config XMUX_ConfigTable;
    XMUX_ConfigTable.DeviceId = 1;
    int init;
    FILE *fp; 
    fp = fopen("trace.txt", "w");
    if (fp == NULL)                 
    {
        perror("File open failed!\n");
    }

    int HW;
    if ((HW = open("/dev/mem", O_RDWR)) < 0) // open Zynq memory
    {
        perror("open failed\n");
    }

    XMUX_ConfigTable.BaseAddr =(u32)mmap(NULL, 16 * sysconf(_SC_PAGESIZE), PROT_READ | PROT_WRITE, MAP_SHARED, HW, 0x43C10000); // 64K
    init = XMUX_CfgInitialize(&mux_inst, &XMUX_ConfigTable);
    XMUX_ConfigTable.Depth = 8;
    XMUX_ConfigTable.Count = 4;
    while (1)
    {
        fprintf(fp,"%ld\n", XMUX_ReadData(mux_inst.Config.BaseAddr));
    }
}
