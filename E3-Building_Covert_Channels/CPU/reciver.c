#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <stdlib.h>

#define DATA_OFFSET 0x00

#define Xil_In32(Addr)  (*(volatile u32 *)(Addr))
#define XTDC_ReadReg(addr, offset) \
    Xil_In32((addr) + (offset))
#define XTDC_DATA_OFFSET 0x00
#define XTDC_ReadData(BaseAddr) \
    XTDC_ReadReg((BaseAddr), XTDC_DATA_OFFSET)
    
typedef uint32_t u32;
int bandwidth;
int trace[10000000];
#define CORE_ATTACKER 3

void pin_to_core(int core)
{
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);

  pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

static char *progname;
int usage(void)
{
   printf("%s: [cycles]\n", progname);
   return 1;
}

int main(int argc, char *argv[])
{
    pin_to_core(CORE_ATTACKER);
    int cycles;
    progname = argv[0];
    if (argc < 2)
       return usage();

    if (sscanf(argv[1], "%d", &cycles) != 1)
       return usage();
    if (sscanf(argv[2], "%d", &bandwidth) != 1)
       return usage();
    FILE *fp; 
    fp = fopen("trace.txt", "w");
    struct timeval start, end;
    if (fp == NULL)                    //判断文件是否成功打开
    {
        perror("File open failed!\n");
    }

    int HW;
    if ((HW = open("/dev/mem", O_RDWR)) < 0) // open Zynq memory
    {
        perror("open failed\n");
    }
    int total = bandwidth*20*cycles;
    int per = bandwidth*20;
    long baseAddr0 =(long)mmap(NULL, 16 * sysconf(_SC_PAGESIZE), PROT_READ | PROT_WRITE, MAP_SHARED, HW, 0x80000000); // 64K
    long baseAddr1 =(long)mmap(NULL, 16 * sysconf(_SC_PAGESIZE), PROT_READ | PROT_WRITE, MAP_SHARED, HW, 0x80001000); // 64K
    long baseAddr2 =(long)mmap(NULL, 16 * sysconf(_SC_PAGESIZE), PROT_READ | PROT_WRITE, MAP_SHARED, HW, 0x80002000); // 64K
    long baseAddr3 =(long)mmap(NULL, 16 * sysconf(_SC_PAGESIZE), PROT_READ | PROT_WRITE, MAP_SHARED, HW, 0x80003000); // 64K
    for(int num = 0; num < total; num++)
    {
        gettimeofday(&start, NULL);
	u32 temp0 = XTDC_ReadData(baseAddr0);
	u32 temp1 = XTDC_ReadData(baseAddr1);
	u32 temp2 = XTDC_ReadData(baseAddr2);
	u32 temp3 = XTDC_ReadData(baseAddr3);
        trace[num] = temp0+temp1+temp2+temp3;
	if((num+1)%per==0){
	    printf("receive bit!\n");
	}
        gettimeofday(&end, NULL);
	while ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)  < 50)  //sampling every 100 us
         {
            gettimeofday(&end, NULL);
         }
    }
    for(int num = 0;num<total;num++){
       fprintf(fp,"%d\n", trace[num]);
    }
    FILE *fp1; 
    fp1 = fopen("receive_signal", "w"); 
    double decode[1001];
    double sum = 0;
    double thresh = 0;
    int index = 0;
    for(int i=0;i<total;i++){
        sum+=trace[i];
        if((i+1)%per==0){
           decode[index] = sum/per;
           thresh += decode[index];
           sum = 0;
           index++;
        }
    }
    thresh /= cycles;
    printf("thresh is %lf",thresh);
    FILE *fp2;
    fp2 = fopen("decode_raw", "w");
    for(int num = 0;num<cycles;num++){
       fprintf(fp1,"%d", decode[num]>thresh?1:0);
       fprintf(fp2,"%lf\n",decode[num]);
    }
    munmap((void*)baseAddr0, 16 * sysconf(_SC_PAGESIZE));
    munmap((void*)baseAddr1, 16 * sysconf(_SC_PAGESIZE));
    munmap((void*)baseAddr2, 16 * sysconf(_SC_PAGESIZE));
    munmap((void*)baseAddr3, 16 * sysconf(_SC_PAGESIZE));
    close(HW);
    fclose(fp);
    fclose(fp1);
    return 0;
}



