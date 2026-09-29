/* canonir -- individualization-refinement canonicalizer for tensor monomials.
 * New code (not derived from xperm.c). C11, plain structs/ints only (FFI friendly).
 *
 * Index code:  CI_IDX(label, up) = 2*label + up.  label > 0 for free indices
 * (must be unique), any label in [-CI_MAX_LABEL, CI_MAX_LABEL] for dummies.
 * A label occurring twice (once Up, once Down) is a dummy pair.
 * Canonical output: dummies are renamed -1,-2,... in order of first appearance
 * (first occurrence Up, second Down); free labels/positions are unchanged.
 */
#ifndef S_CANONIR_H
#define S_CANONIR_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CI_MAX_SLOTS   128
#define CI_MAX_FACTORS 64
#define CI_MAX_RANK    8
#define CI_MAX_SPECS   4
#define CI_MAX_TYPES   64
#define CI_MAX_LABEL   2047

enum { CI_SYM = 1, CI_ANTI = 2, CI_RIEMANN = 3 };
/* Slot-symmetry spec.  slots are 0-based slot ordinals.
 * CI_SYM/CI_ANTI: fully (anti)symmetric in the listed slots (nslots >= 2).
 * CI_RIEMANN: nslots == 4, slots (a,b,c,d): antisym (a b), antisym (c d),
 *             symmetric under pair exchange (a b)<->(c d).  No Bianchi.
 * Specs of one type must have disjoint slot sets.  No specs == NoSym.  */
typedef struct { int kind; int nslots; int slots[CI_MAX_RANK]; } ci_symspec;

typedef struct ci_registry  ci_registry;
typedef struct ci_workspace ci_workspace;

typedef struct {
    int     nfactors;
    int     nslots;                    /* == sum of ranks of the factors   */
    int32_t tid[CI_MAX_FACTORS];       /* tensor type id per factor        */
    int32_t idx[CI_MAX_SLOTS];         /* concatenated slot index codes    */
} ci_monomial;

#define CI_IDX(label, up) ((int32_t)(2 * (label) + ((up) ? 1 : 0)))
#define CI_LABEL(c)       ((int)((c) >> 1))
#define CI_UP(c)          ((int)((c) & 1))

typedef struct { int V, E; long nodes, leaves, autos; } ci_stats;

/* return codes of ci_canonicalize: -1,0,+1 = sign; >=2 = error */
#define CI_ERR_INPUT    2
#define CI_ERR_OVERFLOW 3

ci_registry *ci_registry_new(void);
void         ci_registry_free(ci_registry *);
/* returns type id >= 0, or -1 on invalid spec.  odd = Grassmann-odd. */
int          ci_register_type(ci_registry *, int rank, int odd,
                              const ci_symspec *specs, int nspecs);
int          ci_type_rank(const ci_registry *, int tid);

ci_workspace *ci_workspace_new(void);      /* single allocation, reusable */
void          ci_workspace_free(ci_workspace *);

/* x == sign * out.  sign==0 => monomial vanishes (out left empty). */
int ci_canonicalize(ci_workspace *, const ci_registry *,
                    const ci_monomial *in, ci_monomial *out);
const ci_stats *ci_workspace_stats(const ci_workspace *);

#ifdef __cplusplus
}
#endif
#endif
