/* Quick-and-dirty: one hard instance, three solvers (Sonnet IR, Opus IR, xperm.c Butler-Portugal).
 * Instance: (R_abcd R^abcd)^m  -- 2m identical, freely exchangeable Riemann tensors, n = 8m slots.
 * usage: hard <m> <reps>
 */
#define main xcheck_main
#include "xcheck.c"
#undef main
#include <signal.h>
#include <unistd.h>

void schreier_sims(int *base, int bl, int *GS, int m, int n,
        int *newbase, int *nbl, int **newGS, int *nm, int *num);
void canonical_perm(int *PERM, int SGSQ, int *base, int bl, int *GS, int m, int n,
        int *freeps, int fl, int *dummyps, int dpl, int ob, int metricQ, int *CPERM);

/* build xperm input for a mono whose factors are all Riemann (type 0), all indices dummies */
static int NP; static int GSg[8192*4]; static int ngen;
static void add_gen(int *g){ memcpy(&GSg[ngen*NP], g, sizeof(int)*NP); ngen++; }
static void build_gens(int nf){           /* slot group: per-Riemann syms + adjacent block swaps */
  int n=4*nf; NP=n+2; ngen=0; int g[1024];
  for(int f=0;f<nf;f++){ int s=4*f+1;
    for(int i=1;i<=NP;i++) g[i-1]=i; g[s-1]=s+1; g[s]=s;   g[n]=n+2; g[n+1]=n+1; add_gen(g);   /* (s s+1) antisym */
    for(int i=1;i<=NP;i++) g[i-1]=i; g[s+1]=s+3; g[s+2]=s+2; g[n]=n+2; g[n+1]=n+1; add_gen(g); /* (s+2 s+3) antisym */
    for(int i=1;i<=NP;i++) g[i-1]=i; g[s-1]=s+2; g[s+1]=s; g[s]=s+3; g[s+2]=s+1; add_gen(g);   /* (s s+2)(s+1 s+3) */
  }
  for(int f=0;f+1<nf;f++){ for(int i=1;i<=NP;i++) g[i-1]=i;                                    /* block swap f <-> f+1 */
    for(int j=0;j<4;j++){ g[4*f+j]=4*(f+1)+j+1; g[4*(f+1)+j]=4*f+j+1; } add_gen(g); }
}
/* xperm convention (verified empirically): PERM[name-1] = slot; dummy pair j = names (2j+1 up, 2j+2 down),
   dummyps = slots of (up, down) of each pair */
static void build_perm(const mono *x, int *perm, int *dummyps, int *dpl){
  int n=x->ns; int labs[128], nl=0;
  for(int i=0;i<n;i++){ int seen=0; for(int k=0;k<nl;k++) if(labs[k]==x->lab[i]) seen=1; if(!seen) labs[nl++]=x->lab[i]; }
  for(int i=0;i<nl;i++) for(int k=i+1;k<nl;k++) if(labs[k]<labs[i]){int t=labs[i];labs[i]=labs[k];labs[k]=t;}
  for(int i=0;i<n;i++){ int j=0; while(labs[j]!=x->lab[i]) j++; int name = x->up[i] ? 2*j+1 : 2*j+2; perm[name-1]=i+1; }
  perm[n]=n+1; perm[n+1]=n+2; *dpl=nl; for(int j=0;j<nl;j++){ dummyps[2*j]=perm[2*j]; dummyps[2*j+1]=perm[2*j+1]; }
}
static int xp_sign(const int *c, int n){ if(c[0]==0) return 0; return c[n]==n+1 ? 1 : -1; }

static void on_alarm(int s){ (void)s; printf("TIMEOUT\n"); fflush(stdout); _exit(3); }

int main(int argc, char **argv){
  int m=atoi(argv[1]), reps=argc>2?atoi(argv[2]):20; setup(); rs=12345; setvbuf(stdout,NULL,_IONBF,0);
  mono base; kretsch(&base,m); int nf=2*m, n=8*m;
  mono Y[64]; for(int i=0;i<64;i++) equiv(&base,&Y[i]);
  build_gens(nf);
  printf("# instance (R_abcd R^abcd)^%d: %d identical Riemanns, n=%d slots, %d dummy pairs; slot group order 8^%d * %d!\n", m, nf, n, n/2, nf, nf);
  printf("# xperm: %d generators on %d points\n", ngen, NP);
  signal(SIGALRM, on_alarm);

  /* ---- correctness gate: all 64 relabellings -> same canonical form per solver, consistent relative signs ---- */
  int c0[256], c[256], perm[256], dps[256], dpl, bs[256]; for(int i=0;i<n;i++) bs[i]=i+1;
  res r0; run(&Y[0],&r0); int sx0;
  alarm(300);
  double t0=now(); build_perm(&Y[0],perm,dps,&dpl); canonical_perm(perm,0,bs,n,GSg,ngen,NP,NULL,0,dps,dpl,0,1,c0); double tfirst=now()-t0;
  alarm(0);
  sx0=xp_sign(c0,n);
  printf("# first xperm call (incl. Schreier-Sims): %.3f ms, sign %d; Sonnet sign %d, Opus sign %d\n", tfirst*1e3, sx0, r0.s, r0.o);
  int bad=0; int slow = tfirst>1.0;
  if(slow) printf("# xperm too slow for gate/timing loops; S vs O only below\n");
  for(int i=1;i<64;i++){ if(slow){ res r; run(&Y[i],&r); if(!eqS(&r.so,&r0.so)||!eqO(&r.oo,&r0.oo)||r.s*r0.s!=r.o*r0.o) bad++; continue; } res r; run(&Y[i],&r); build_perm(&Y[i],perm,dps,&dpl); alarm(600);
    canonical_perm(perm,0,bs,n,GSg,ngen,NP,NULL,0,dps,dpl,0,1,c); alarm(0); int sx=xp_sign(c,n);
    int sameX = !memcmp(c,c0,sizeof(int)*NP) || (c[0]!=0 && !memcmp(c,c0,sizeof(int)*n));
    if(!sameX || !eqS(&r.so,&r0.so) || !eqO(&r.oo,&r0.oo)) { bad++; printf("form mismatch at %d (x %d S %d O %d)\n", i, sameX, eqS(&r.so,&r0.so), eqO(&r.oo,&r0.oo)); }
    if(sx*sx0 != r.s*r0.s || r.s*r0.s != r.o*r0.o){ bad++; printf("relative sign mismatch at %d: xperm %d Sonnet %d Opus %d\n", i, sx*sx0, r.s*r0.s, r.o*r0.o); }
  }
  printf("# correctness gate over 64 relabellings: %s\n", bad?"FAILED":"all three consistent");

  /* ---- xperm with precomputed SGS (cached per factor pattern) ---- */
  int nb[256], nbl, nm, num; int *nGS=NULL;
  if(tfirst<=1.0) { int *tmp=malloc(sizeof(int)*ngen*NP); memcpy(tmp,GSg,sizeof(int)*ngen*NP);
    nGS=malloc(sizeof(int)*ngen*NP); memcpy(nGS,tmp,sizeof(int)*ngen*NP); free(tmp);
    double a=now(); schreier_sims(bs,n,GSg,ngen,NP,nb,&nbl,&nGS,&nm,&num); printf("# Schreier-Sims once: %.3f ms (SGS: %d gens, base %d)\n",(now()-a)*1e3,nm,nbl); }

  /* ---- timing: median over batches of all 64 relabellings ---- */
  ci_monomial sa[64]; canonir_monomial ob[64]; int P[64][256], D[64][256], DL[64];
  for(int i=0;i<64;i++){ toS(&Y[i],&sa[i]); toO(&Y[i],&ob[i]); build_perm(&Y[i],P[i],D[i],&DL[i]); }
  double tS[15],tO[15],tX[15],tXc[15]; ci_monomial so; canonir_monomial oo; volatile int sink=0;
  int xreps = tfirst>0.05 ? 1 : reps;
  for(int b=0;b<15;b++){
    double a=now(); for(int r=0;r<reps;r++) for(int i=0;i<64;i++) sink+=ci_canonicalize(wsS,regS,&sa[i],&so); tS[b]=(now()-a)/(reps*64);
    a=now(); for(int r=0;r<reps;r++) for(int i=0;i<64;i++) sink+=canonir_canonicalize(regO,wsO,&ob[i],&oo); tO[b]=(now()-a)/(reps*64);
    if(slow){ tXc[b]=tX[b]=tfirst; continue; }
    a=now(); for(int r=0;r<xreps;r++) for(int i=0;i<64;i++){ canonical_perm(P[i],1,nb,nbl,nGS,nm,NP,NULL,0,D[i],DL[i],0,1,c); sink+=c[0]; } tXc[b]=(now()-a)/(xreps*64);
    if(b<5){ a=now(); for(int i=0;i<64;i++){ canonical_perm(P[i],0,bs,n,GSg,ngen,NP,NULL,0,D[i],DL[i],0,1,c); sink+=c[0]; } tX[b]=(now()-a)/64; } else tX[b]=tX[b%5];
  }
  qsort(tS,15,sizeof(double),cmpd); qsort(tO,15,sizeof(double),cmpd); qsort(tX,15,sizeof(double),cmpd); qsort(tXc,15,sizeof(double),cmpd);
  printf("RESULT m=%d n=%d  Sonnet %10.1f us  Opus %10.1f us  xperm(cached SGS) %10.1f us  xperm(stock, SGS per call) %10.1f us\n",
         m, n, tS[7]*1e6, tO[7]*1e6, tXc[7]*1e6, tX[7]*1e6);
  (void)sink; return bad?1:0;
}
