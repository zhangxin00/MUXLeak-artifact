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

#define M0 10000

#define CORE_VICTIM0 0
#define CORE_VICTIM1 1
#define CORE_VICTIM2 2

int bandwidth;  // ms

void pin_to_core(int core)
{
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);

  pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

void* stress_fun1(void* dummy)
{
    pin_to_core(CORE_VICTIM1);
    struct timeval start, end;
    int j = 0;
    gettimeofday(&start, NULL);
    gettimeofday(&end, NULL);
    while ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000 < ((float)bandwidth*0.9))
    {
       for (j = 0; j < M0; j++)
       {
          sqrt((double)rand());
       }
       gettimeofday(&end, NULL);
    }
    pthread_exit(NULL);
}

void* stress_fun2(void* dummy)
{
    pin_to_core(CORE_VICTIM2);
    struct timeval start, end;
    int j = 0;
    gettimeofday(&start, NULL);
    gettimeofday(&end, NULL);
    while ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000 < ((float)bandwidth*0.9))
    {
       for (j = 0; j < M0; j++)
       {
          sqrt((double)rand());
       }
       gettimeofday(&end, NULL);
    }
    pthread_exit(NULL);
}

static char *progname;
int usage(void)
{
   printf("%s: [cycles]\n", progname);
   return 1;
}

int main(int argc, char *argv[])
{
   pin_to_core(CORE_VICTIM0);
   int cycles;
   progname = argv[0];
   if (argc < 2)
      return usage();

   if (sscanf(argv[1], "%d", &cycles) != 1)
      return usage();
   if (sscanf(argv[2], "%d", &bandwidth) != 1)
      return usage();
   int str[cycles + 1];
   memset(str, 0, cycles);

   int i, j, k, count;
   double time0;
   struct timeval start, end;

   for (i = 0; i < cycles; i++)
   {
      str[i] = rand() % 2;
   }

   str[0]=1;

   for (i = 0; i < cycles; i++)
   {
      gettimeofday(&start, NULL);
      gettimeofday(&end, NULL);
      if (str[i] == 1)
      {
	    pthread_t p1;
	    pthread_create(&p1,NULL, stress_fun1, NULL);
	    pthread_t p2;
	    pthread_create(&p2,NULL, stress_fun2, NULL);
            while ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000 < ((float)bandwidth*0.9))
            {
               for (j = 0; j < M0; j++)
               {
                  sqrt((double)rand());
               }
               gettimeofday(&end, NULL);
            }
         printf("i=%d,send 1! \n", i);
         while ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000 < bandwidth)
         {
            gettimeofday(&end, NULL);
         }
      }
      else if (str[i] == 0)
      {

         usleep(bandwidth*1000);
         printf("i=%d,send 0! \n", i);
	 /*
         while ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000 < bandwidth)
         {
            gettimeofday(&end, NULL);
         }
	 */
      }
   }
   FILE *fp_write = fopen("./send_signal", "w");
   for (i = 0; i < cycles; i++)
   {
      // printf("%d",str[i]);
      fprintf(fp_write, "%d", str[i]);
   }
   printf("finish sending\n");

   fclose(fp_write);

   return 0;
}




