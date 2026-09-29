/* Blind cross-check of two independent canonir implementations on identical inputs.
 * S = Sonnet (ci_*), O = Opus (canonir_*).
 * usage: xcheck check <nmono> <maxslots> <seed>
 *        xcheck bench <seed>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "s_canonir.h"
#include "o_canonir.h"

static uint64_t rs; static int corrupt=0;
static uint64_t xr(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static int rnd(int n){ return (int)(xr()%(uint64_t)n); }

/* ---- common type table ---- */
typedef struct { int rank, odd, nspec; int kind[3], ns[3], sl[3][6]; } ttype;
static const ttype T[] = {
  {4,0,1,{3},{4},{{0,1,2,3}}},              /* 0 Riem */
  {2,0,1,{1},{2},{{0,1}}},                  /* 1 S sym */
  {2,0,1,{2},{2},{{0,1}}},                  /* 2 A antisym */
  {3,0,0,{0},{0},{{0}}},                    /* 3 X nosym */
  {4,0,2,{1,2},{2,2},{{0,1},{2,3}}},        /* 4 W sym(01) anti(23) */
  {1,1,0,{0},{0},{{0}}},                    /* 5 psi odd vector */
  {2,1,1,{2},{2},{{0,1}}},                  /* 6 chi odd antisym */
  {3,0,1,{1},{3},{{0,1,2}}},                /* 7 fully sym 3 */
  {3,0,1,{2},{3},{{0,1,2}}},                /* 8 fully antisym 3 */
  {4,0,0,{0},{0},{{0}}},                    /* 9 Y nosym 4 */
  {1,0,0,{0},{0},{{0}}},                    /* 10 V vector */
  {6,0,2,{3,1},{4,2},{{0,1,2,3},{4,5}}},    /* 11 Riem x sym pair */
  {6,0,2,{1,1},{2,4},{{0,1},{2,3,4,5}}},    /* 12 d4h-like */
  {3,1,1,{2},{3},{{0,1,2}}},                /* 13 odd fully antisym 3 */
};
#define NT ((int)(sizeof T/sizeof T[0]))
static int stid[NT];

typedef struct { int nf, ns; int ty[64]; int lab[128], up[128]; } mono;

static ci_registry *regS; static ci_workspace *wsS;
static canonir_registry *regO; static canonir_workspace *wsO;

static void setup(void){
  regS = ci_registry_new(); wsS = ci_workspace_new();
  regO = canonir_registry_new(); wsO = canonir_workspace_new();
  for (int t=0;t<NT;t++){
    ci_symspec sp[3]; memset(sp,0,sizeof sp);
    for (int s=0;s<T[t].nspec;s++){ sp[s].kind=T[t].kind[s]; sp[s].nslots=T[t].ns[s];
      for(int j=0;j<T[t].ns[s];j++) sp[s].slots[j]=T[t].sl[s][j]; }
    stid[t] = ci_register_type(regS, T[t].rank, T[t].odd, sp, T[t].nspec);
    if (stid[t]<0){ fprintf(stderr,"S register fail %d\n",t); exit(2);}
    if (canonir_define_type(regO, t, T[t].rank, T[t].odd)!=0){ fprintf(stderr,"O define fail %d\n",t); exit(2);}
    for (int s=0;s<T[t].nspec;s++){ int32_t sl[6]; for(int j=0;j<T[t].ns[s];j++) sl[j]=T[t].sl[s][j];
      if (canonir_type_add_sym(regO, t, T[t].kind[s], T[t].ns[s], sl)!=0){ fprintf(stderr,"O sym fail %d\n",t); exit(2);} }
  }
}

static void toS(const mono *m, ci_monomial *o){
  memset(o,0,sizeof *o); o->nfactors=m->nf; o->nslots=m->ns;
  for(int f=0;f<m->nf;f++) o->tid[f]=stid[m->ty[f]];
  for(int i=0;i<m->ns;i++) o->idx[i]=CI_IDX(m->lab[i], m->up[i]);
}
static void toO(const mono *m, canonir_monomial *o){
  canonir_mono_clear(o); int p=0;
  for(int f=0;f<m->nf;f++){ canonir_index ix[32]; int r=T[m->ty[f]].rank;
    for(int j=0;j<r;j++){ ix[j].label=m->lab[p+j]; ix[j].pos=m->up[p+j]?CANONIR_UP:CANONIR_DOWN; }
    if (canonir_mono_push(regO,o,m->ty[f],ix)!=0){ fprintf(stderr,"O push fail\n"); exit(2);} p+=r; }
}
/* convert outputs back to common mono (type ids -> table ids) */
static int sid2t(int s){ for(int t=0;t<NT;t++) if(stid[t]==s) return t; return -1; }
static void fromS(const ci_monomial *o, mono *m){ m->nf=o->nfactors; m->ns=o->nslots;
  for(int f=0;f<m->nf;f++) m->ty[f]=sid2t(o->tid[f]);
  for(int i=0;i<m->ns;i++){ m->lab[i]=CI_LABEL(o->idx[i]); m->up[i]=CI_UP(o->idx[i]); } }
static void fromO(const canonir_monomial *o, mono *m){ m->nf=o->nfactors; m->ns=o->nslots;
  for(int f=0;f<m->nf;f++) m->ty[f]=o->type[f];
  for(int i=0;i<m->ns;i++){ m->lab[i]=o->idx[i].label; m->up[i]=o->idx[i].pos==CANONIR_UP; } }

static int eqS(const ci_monomial *a, const ci_monomial *b){
  if(a->nfactors!=b->nfactors||a->nslots!=b->nslots) return 0;
  return !memcmp(a->tid,b->tid,sizeof(int32_t)*a->nfactors) && !memcmp(a->idx,b->idx,sizeof(int32_t)*a->nslots); }
static int eqO(const canonir_monomial *a, const canonir_monomial *b){
  if(a->nfactors!=b->nfactors||a->nslots!=b->nslots) return 0;
  if(memcmp(a->type,b->type,sizeof(int32_t)*a->nfactors)) return 0;
  for(int i=0;i<a->nslots;i++) if(a->idx[i].label!=b->idx[i].label||a->idx[i].pos!=b->idx[i].pos) return 0;
  return 1; }

/* random monomial: small palette of types to force identical factors */
static void gen(mono *m, int maxslots){
  int pal[3]; int np=1+rnd(3); for(int i=0;i<np;i++) pal[i]=rnd(NT);
  m->nf=0; m->ns=0;
  int target = 2+rnd(maxslots-1);
  while(m->ns < target && m->nf<64){ int t=pal[rnd(np)]; if(m->ns+T[t].rank>maxslots) break;
    m->ty[m->nf++]=t; m->ns+=T[t].rank; }
  if(m->ns==0){ m->ty[0]=10; m->nf=1; m->ns=1; }
  int nfree = (m->ns%2) ? 1+2*rnd(2) : 2*rnd(2); if(nfree>m->ns) nfree=m->ns%2;
  int perm[128]; for(int i=0;i<m->ns;i++) perm[i]=i;
  for(int i=m->ns-1;i>0;i--){ int j=rnd(i+1); int t=perm[i]; perm[i]=perm[j]; perm[j]=t; }
  int k=0; for(int i=0;i<nfree;i++,k++){ m->lab[perm[k]]=1+i; m->up[perm[k]]=rnd(2); }
  int d=0; for(;k+1<m->ns;k+=2,d++){ int u=rnd(2); m->lab[perm[k]]=100+d; m->up[perm[k]]=u; m->lab[perm[k+1]]=100+d; m->up[perm[k+1]]=!u; }
}
/* equivalent (up to sign) transform */
static void equiv(const mono *x, mono *y){
  int nf=x->nf, off[65]; off[0]=0; for(int f=0;f<nf;f++) off[f+1]=off[f]+T[x->ty[f]].rank;
  int ord[64]; for(int f=0;f<nf;f++) ord[f]=f; for(int i=nf-1;i>0;i--){int j=rnd(i+1);int t=ord[i];ord[i]=ord[j];ord[j]=t;}
  y->nf=nf; y->ns=x->ns; int p=0;
  for(int q=0;q<nf;q++){ int f=ord[q], t=x->ty[f], r=T[t].rank; y->ty[q]=t;
    int sl[32]; for(int j=0;j<r;j++) sl[j]=j;
    for(int rep=0;rep<3;rep++) for(int s=0;s<T[t].nspec;s++){ if(rnd(2)) continue;
      const int *b=T[t].sl[s]; int kind=T[t].kind[s];
      if(kind==3){ int c=rnd(3); if(c==0){int u=sl[b[0]];sl[b[0]]=sl[b[1]];sl[b[1]]=u;}
        else if(c==1){int u=sl[b[2]];sl[b[2]]=sl[b[3]];sl[b[3]]=u;}
        else {int u=sl[b[0]];sl[b[0]]=sl[b[2]];sl[b[2]]=u; u=sl[b[1]];sl[b[1]]=sl[b[3]];sl[b[3]]=u;} }
      else { int n=T[t].ns[s]; int i=rnd(n), j=rnd(n); int u=sl[b[i]]; sl[b[i]]=sl[b[j]]; sl[b[j]]=u; } }
    for(int j=0;j<r;j++){ y->lab[p+j]=x->lab[off[f]+sl[j]]; y->up[p+j]=x->up[off[f]+sl[j]]; } p+=r; }
  /* rename dummies, random up/down swap per pair */
  int map[4096]; memset(map,-1,sizeof map); int flip[4096]; int next=500+rnd(50);
  for(int i=0;i<y->ns;i++){ int l=y->lab[i]; if(l>=100){ if(map[l]<0){ map[l]=next; next+=1+rnd(3); flip[l]=rnd(2);} y->lab[i]=map[l]; if(flip[l]) y->up[i]=!y->up[i]; } }
}
/* likely-inequivalent: re-pair dummies */
static void repair(const mono *x, mono *z){ *z=*x; int pos[128], n=0;
  for(int i=0;i<z->ns;i++) if(z->lab[i]>=100) pos[n++]=i;
  for(int i=n-1;i>0;i--){int j=rnd(i+1);int t=pos[i];pos[i]=pos[j];pos[j]=t;}
  for(int k=0;k+1<n;k+=2){ int u=rnd(2); z->lab[pos[k]]=100+k/2; z->up[pos[k]]=u; z->lab[pos[k+1]]=100+k/2; z->up[pos[k+1]]=!u; } }

typedef struct { int s; ci_monomial so; int o; canonir_monomial oo; } res;
static void run(const mono *m, res *r){ ci_monomial a; canonir_monomial b; toS(m,&a); toO(m,&b);
  r->s=ci_canonicalize(wsS,regS,&a,&r->so); r->o=canonir_canonicalize(regO,wsO,&b,&r->oo);
  if(corrupt && r->s!=0 && (xr()%corrupt)==0) r->s=-r->s; }

static long nfail=0, nchk=0, ninequiv=0, nequivr=0;
#define CHK(c, ...) do{ nchk++; if(!(c)){ if(nfail<20){ fprintf(stderr,"FAIL: " __VA_ARGS__); fprintf(stderr,"\n"); } nfail++; } }while(0)
static void dump(const char*tag,const mono*m){ fprintf(stderr,"  %s:",tag); int p=0; for(int f=0;f<m->nf;f++){ fprintf(stderr," T%d[",m->ty[f]); for(int j=0;j<T[m->ty[f]].rank;j++,p++) fprintf(stderr,"%s%d",m->up[p]?"^":"_",m->lab[p]); fprintf(stderr,"]"); } fprintf(stderr,"\n"); }

static void pair_check(const mono *x, const res *rx, const mono *y, const res *ry, const char *kind){
  CHK((rx->s==0)==(rx->o==0), "zero disagree x (%s) S=%d O=%d", kind, rx->s, rx->o);
  CHK((ry->s==0)==(ry->o==0), "zero disagree y (%s) S=%d O=%d", kind, ry->s, ry->o);
  if(rx->s==0||ry->s==0||rx->o==0||ry->o==0) return;
  int es=eqS(&rx->so,&ry->so), eo=eqO(&rx->oo,&ry->oo);
  if(es!=eo){ CHK(0, "equivalence disagree (%s): S says %d, O says %d", kind, es, eo); dump("x",x); dump("y",y); return; }
  nchk++; if(!strcmp(kind,"repair")){ if(es) nequivr++; else ninequiv++; }
  if(es) CHK(rx->s*ry->s == rx->o*ry->o, "relative sign disagree (%s)", kind);
}
/* absolute cross-feed: x = sS * outS; canonO(outS) must equal canonO(x) with sO(outS)*sO(x) == sS */
static void cross_feed(const mono *x, const res *r){
  if(r->s==0||r->o==0) return;
  mono ms, mo; fromS(&r->so,&ms); fromO(&r->oo,&mo);
  res a, b; run(&ms,&a); run(&mo,&b);
  CHK(a.o!=0 && eqO(&a.oo,&r->oo), "O(canonS(x)) form != O(x)");
  if(a.o!=0) CHK(a.o * r->o == r->s, "abs sign S->O mismatch: sS=%d sO(x)=%d sO(outS)=%d", r->s, r->o, a.o);
  CHK(b.s!=0 && eqS(&b.so,&r->so), "S(canonO(x)) form != S(x)");
  if(b.s!=0) CHK(b.s * r->s == r->o, "abs sign O->S mismatch");
  (void)x;
}

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y; }

static void chain(mono *m, int k){ m->nf=k; m->ns=4*k; for(int i=0;i<k;i++){ m->ty[i]=0;
  int a=2*i, b=2*i+1, c=(2*i+2)%(2*k), d=(2*i+3)%(2*k);
  m->lab[4*i]=100+a; m->up[4*i]=0; m->lab[4*i+1]=100+b; m->up[4*i+1]=0; m->lab[4*i+2]=100+c; m->up[4*i+2]=1; m->lab[4*i+3]=100+d; m->up[4*i+3]=1; } }
static void kretsch(mono *m, int p){ m->nf=2*p; m->ns=8*p; for(int i=0;i<p;i++){ for(int j=0;j<4;j++){
  m->ty[2*i]=0; m->ty[2*i+1]=0; m->lab[8*i+j]=100+4*i+j; m->up[8*i+j]=0; m->lab[8*i+4+j]=100+4*i+j; m->up[8*i+4+j]=1; } } }

static void bench_set(const char *name, mono *set, int n, int reps){
  ci_monomial *sa=malloc(sizeof(ci_monomial)*n); canonir_monomial *ob=malloc(sizeof(canonir_monomial)*n);
  for(int i=0;i<n;i++){ toS(&set[i],&sa[i]); toO(&set[i],&ob[i]); }
  ci_monomial so; canonir_monomial oo; double ts[31], to[31]; volatile int sink=0;
  for(int w=0;w<n;w++){ sink+=ci_canonicalize(wsS,regS,&sa[w],&so); sink+=canonir_canonicalize(regO,wsO,&ob[w],&oo); }
  for(int b=0;b<31;b++){
    double t0=now(); for(int r=0;r<reps;r++) for(int i=0;i<n;i++) sink+=ci_canonicalize(wsS,regS,&sa[i],&so); double t1=now();
    for(int r=0;r<reps;r++) for(int i=0;i<n;i++) sink+=canonir_canonicalize(regO,wsO,&ob[i],&oo); double t2=now();
    if(b&1){ ts[b]=(t1-t0)/(reps*n)*1e9; to[b]=(t2-t1)/(reps*n)*1e9; }
    else {  /* alternate order */
      double u0=now(); for(int r=0;r<reps;r++) for(int i=0;i<n;i++) sink+=canonir_canonicalize(regO,wsO,&ob[i],&oo); double u1=now();
      for(int r=0;r<reps;r++) for(int i=0;i<n;i++) sink+=ci_canonicalize(wsS,regS,&sa[i],&so); double u2=now();
      to[b]=(u1-u0)/(reps*n)*1e9; ts[b]=(u2-u1)/(reps*n)*1e9; }
  }
  qsort(ts,31,sizeof(double),cmpd); qsort(to,31,sizeof(double),cmpd);
  printf("%-34s n=%5d  Sonnet %9.0f ns  Opus %9.0f ns  S/O = %5.2f\n", name, n, ts[15], to[15], ts[15]/to[15]);
  free(sa); free(ob); (void)sink;
}

int main(int argc, char **argv){
  setup();
  if(argc>=5 && !strcmp(argv[1],"check")){
    int N=atoi(argv[2]), maxs=atoi(argv[3]); rs=strtoull(argv[4],0,10)|1; if(getenv("CORRUPT")) corrupt=atoi(getenv("CORRUPT"));
    long zeros=0, errs=0, eqpairs=0;
    for(int it=0;it<N;it++){
      mono x,y,z; gen(&x,maxs); equiv(&x,&y); repair(&x,&z);
      res rx,ry,rz; run(&x,&rx); run(&y,&ry); run(&z,&rz);
      if(rx.s>1||rx.s<-1||rx.o<-1||ry.s>1||ry.o<-1||rz.s>1||rz.o<-1){ errs++; if(errs<5){fprintf(stderr,"error code S=%d O=%d\n",rx.s,rx.o); dump("x",&x);} continue; }
      if(rx.s==0) zeros++;
      pair_check(&x,&rx,&y,&ry,"equiv");
      /* y is equivalent to x: both must say so */
      if(rx.s!=0 && ry.s!=0){ CHK(eqS(&rx.so,&ry.so), "S: equivalent pair not identified"); CHK(eqO(&rx.oo,&ry.oo), "O: equivalent pair not identified"); eqpairs++; }
      pair_check(&x,&rx,&z,&rz,"repair");
      cross_feed(&x,&rx);
    }
    printf("check: %d monomials (maxslots=%d, seed=%s): %ld zeros, %ld error-code instances, %ld equivalent pairs; %ld checks, %ld FAILURES; repair pairs: %ld inequivalent, %ld equivalent\n",
           N, maxs, argv[4], zeros, errs, eqpairs, nchk, nfail, ninequiv, nequivr);
    return nfail?1:0;
  }
  if(argc>=3 && !strcmp(argv[1],"bench")){
    rs=strtoull(argv[2],0,10)|1; static mono set[2048];
    int sizes[]={12,24,48};
    for(int si=0;si<3;si++){ for(int i=0;i<1024;i++) gen(&set[i],sizes[si]); char nm[64]; snprintf(nm,64,"random, <=%d slots",sizes[si]); bench_set(nm,set,1024, sizes[si]>24?2:5); }
    int ks[]={3,8,12}; for(int i=0;i<3;i++){ for(int j=0;j<64;j++){ mono c; chain(&c,ks[i]); equiv(&c,&set[j]); } char nm[64]; snprintf(nm,64,"Riemann chain k=%d (64 relabellings)",ks[i]); bench_set(nm,set,64,50); }
    int ps[]={2,4,6}; for(int i=0;i<3;i++){ for(int j=0;j<64;j++){ mono c; kretsch(&c,ps[i]); equiv(&c,&set[j]); } char nm[64]; snprintf(nm,64,"(RabcdR^abcd)^%d (64 relabellings)",ps[i]); bench_set(nm,set,64,20); }
    return 0;
  }
  fprintf(stderr,"usage\n"); return 2;
}
