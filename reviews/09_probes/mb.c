#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <sys/mman.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static uint64_t rng=88172645463325252ull; static uint64_t xs(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
/* dependent-load latency: random cyclic permutation, 64B stride (one pointer per line) */
double chase(size_t bytes){
  size_t n=bytes/64; char*buf=aligned_alloc(2<<20,((n*64+(2<<20)-1)/(2<<20))*(2<<20)); madvise(buf,n*64,MADV_HUGEPAGE); size_t*perm=malloc(n*sizeof(size_t));
  for(size_t i=0;i<n;i++)perm[i]=i; for(size_t i=n-1;i>0;i--){size_t j=xs()%(i+1);size_t t=perm[i];perm[i]=perm[j];perm[j]=t;}
  for(size_t i=0;i<n;i++) *(void**)(buf+perm[i]*64)=buf+perm[(i+1)%n]*64;
  void*p=buf+perm[0]*64; size_t iters=20000000; for(size_t i=0;i<n;i++)p=*(void**)p;
  double t0=now(); for(size_t i=0;i<iters;i++)p=*(void**)p; double t=now()-t0;
  if(p==0)puts(""); free(buf);free(perm); return t/iters*1e9;
}
/* read bandwidth: sum of uint64 over a large array, nthreads */
double readbw(size_t bytes,int nt){
  size_t n=bytes/8; uint64_t*a=aligned_alloc(2<<20,bytes); madvise(a,bytes,MADV_HUGEPAGE);
  #pragma omp parallel for num_threads(nt)
  for(size_t i=0;i<n;i++)a[i]=i;
  double best=0; for(int r=0;r<5;r++){ uint64_t s=0; double t0=now();
    #pragma omp parallel for num_threads(nt) reduction(+:s)
    for(size_t i=0;i<n;i++)s+=a[i];
    double t=now()-t0; if(s==1)puts(""); double bw=bytes/t/1e9; if(bw>best)best=bw;}
  free(a); return best;
}
/* random 8B reads into big table with independent addresses (memory-level parallelism) */
double randrd(size_t bytes,int nt){
  size_t n=bytes/8; uint64_t*a=aligned_alloc(2<<20,bytes); madvise(a,bytes,MADV_HUGEPAGE); memset(a,1,bytes); size_t N=40000000; double t0=now(); uint64_t s=0;
  #pragma omp parallel num_threads(nt) reduction(+:s)
  { uint64_t r=0x9E3779B97F4A7C15ull*(omp_get_thread_num()+1);
    #pragma omp for
    for(size_t i=0;i<N;i++){ r^=r<<13;r^=r>>7;r^=r<<17; s+=a[r%n]; } }
  double t=now()-t0; if(s==1)puts(""); free(a); return t/N*1e9; /* ns per access, aggregate */
}
int main(){
  size_t sz[]={16<<10,32<<10,256<<10,1<<20,4<<20,8<<20,64<<20,512<<20};
  printf("# latency (dependent loads), ns\n");
  for(int i=0;i<8;i++)printf("%8zu KiB  %6.2f ns\n",sz[i]>>10,chase(sz[i]));
  printf("# read bandwidth 1 GiB, GB/s\n");
  int nts[]={1,2,4,12}; for(int i=0;i<4;i++)printf("threads=%2d  %6.1f GB/s\n",nts[i],readbw(1ul<<30,nts[i]));
  printf("# independent random 8B reads in 1 GiB table, aggregate ns/access\n");
  for(int i=0;i<4;i++)printf("threads=%2d  %6.2f ns/access\n",nts[i],randrd(1ul<<30,nts[i]));
}
