/*
 * canonir -- prototype individualization-refinement (IR) canonicalizer for
 * tensor monomials with monoterm slot symmetries, dummy indices and Grassmann
 * parity.  C11, no dependencies, FFI-friendly (plain structs and ints, no
 * callbacks).
 *
 * Semantics:   x == sign * canonical(x)   as tensor expressions,
 *              sign == 0  iff  x vanishes by (monoterm) symmetry.
 *
 * Canonical output conventions
 *   - factors ordered by tensor type id, identical factors in canonical order;
 *   - free indices keep their label AND position (never changed);
 *   - dummy pairs are renamed to labels -1, -2, ..., -d in order of first
 *     appearance in the canonical output; the first occurrence is Up, the
 *     second Down (symmetric metric: this costs no sign);
 *   - when sign == 0 the output monomial is empty (nfactors == 0).
 *
 * Input requirements
 *   - a label occurring once is free and must be >= 0;
 *   - a label occurring twice is a dummy pair and must occur once Up and once
 *     Down (any int32 value, including the negative labels produced as output,
 *     so canonical output can be fed back in);
 *   - no label may occur more than twice.
 *
 * Memory: canonir_canonicalize performs no heap allocation.  All scratch
 * memory lives in a canonir_workspace created once (sized for <= 128 slots).
 * A workspace must not be used by two threads at once; a registry is
 * read-only during canonicalization and may be shared.
 */
#ifndef O_CANONIR_H
#define O_CANONIR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CANONIR_MAX_SLOTS   128  /* total index slots per monomial        */
#define CANONIR_MAX_FACTORS 128  /* factors per monomial                  */
#define CANONIR_MAX_RANK    32   /* slots per tensor type                 */
#define CANONIR_MAX_TYPES   1024 /* tensor type ids are 0..MAX_TYPES-1    */

enum { CANONIR_DOWN = 0, CANONIR_UP = 1 };

/* Slot-symmetry kinds for canonir_type_add_sym. */
enum {
    CANONIR_SYM     = 1, /* fully symmetric in the listed slots (>= 2)          */
    CANONIR_ANTISYM = 2, /* fully antisymmetric in the listed slots (>= 2)      */
    CANONIR_RIEMANN = 3  /* 4 slots (s0,s1,s2,s3): antisym (s0 s1), (s2 s3),
                            symmetric under pair exchange (s0 s1)<->(s2 s3)   */
};

/* Error codes (returned as negative values <= -2 so they never collide with
   the sign values -1/0/+1). */
enum {
    CANONIR_EINVAL   = -2, /* malformed monomial or bad argument            */
    CANONIR_ETYPE    = -3, /* unknown tensor type                           */
    CANONIR_ELIMIT   = -4, /* instance exceeds a compile-time limit          */
    CANONIR_EINDEX   = -5  /* bad index structure (label count, positions)  */
};

typedef struct {
    int32_t label;
    int32_t pos;   /* CANONIR_UP or CANONIR_DOWN */
} canonir_index;

/* A monomial: factors in product order; the slots of factor f are the
   rank(type[f]) consecutive entries of idx starting at the sum of the ranks
   of the previous factors. */
typedef struct {
    int32_t nfactors;
    int32_t nslots;
    int32_t type[CANONIR_MAX_FACTORS];
    canonir_index idx[CANONIR_MAX_SLOTS];
} canonir_monomial;

typedef struct {
    int64_t nodes;          /* search-tree nodes (refinements), incl. root  */
    int64_t leaves;         /* leaves whose certificate was computed        */
    int64_t automorphisms;  /* automorphisms found (generators)             */
    int64_t implicit;       /* ... of which found by implicit_aut (no leaf) */
    int32_t V, E;           /* graph size of the last instance              */
} canonir_stats;

typedef struct canonir_registry canonir_registry;
typedef struct canonir_workspace canonir_workspace;

/* ---- tensor-type registry (setup; may allocate) ---- */
canonir_registry *canonir_registry_new(void);
void canonir_registry_free(canonir_registry *reg);
/* Define (or redefine) a type.  odd != 0 marks a Grassmann-odd tensor. */
int canonir_define_type(canonir_registry *reg, int32_t type_id, int32_t rank, int32_t odd);
/* Add a slot-symmetry block.  Blocks of one type must use disjoint slots.
   Slots are 0-based.  Pairwise Symmetric(i,j)/Antisymmetric(i,j) are
   CANONIR_SYM/CANONIR_ANTISYM with nslots == 2. */
int canonir_type_add_sym(canonir_registry *reg, int32_t type_id, int32_t kind,
                         int32_t nslots, const int32_t *slots);
int32_t canonir_type_rank(const canonir_registry *reg, int32_t type_id); /* -1 if undefined */

/* ---- monomial building helpers ---- */
void canonir_mono_clear(canonir_monomial *m);
/* Append a factor; idx must hold rank(type_id) entries. Returns 0 or error. */
int canonir_mono_push(const canonir_registry *reg, canonir_monomial *m,
                      int32_t type_id, const canonir_index *idx);

/* ---- workspace ---- */
canonir_workspace *canonir_workspace_new(void);
void canonir_workspace_free(canonir_workspace *ws);
void canonir_workspace_stats(const canonir_workspace *ws, canonir_stats *st);

/* ---- canonicalization (hot path; no allocation) ----
   Returns +1 / -1 (x = sign * out), 0 (x vanishes; out is empty), or a
   negative error code <= -2.  in and out may not alias. */
int canonir_canonicalize(const canonir_registry *reg, canonir_workspace *ws,
                         const canonir_monomial *in, canonir_monomial *out);

#ifdef __cplusplus
}
#endif
#endif /* O_CANONIR_H */
