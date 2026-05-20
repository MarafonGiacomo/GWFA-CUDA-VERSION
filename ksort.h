/* The MIT License
   Copyright (c) 2008, 2011 Attractive Chaos <attractor@live.co.uk>
   ... (omesso per brevità; mantieni la licenza completa qui) ...
*/
#ifndef AC_KSORT_H
#define AC_KSORT_H

#include <stdlib.h>
#include <assert.h>

#define RS_MIN_SIZE 64
#define RS_MAX_BITS 6
#define KRADIX_SORT_INIT_D(name, rstype_t, rskey, sizeof_key) \
    typedef struct { \
        rstype_t *b, *e; \
    } rsbucket_##name##_t; \
    \
    void rs_insertsort_##name(rstype_t *beg, rstype_t *end) \
    { \
        rstype_t *i; \
        for (i = beg + 1; i < end; ++i) \
            if (rskey(*i) < rskey(*(i - 1))) { \
                rstype_t *j, tmp = *i; \
                for (j = i; j > beg && rskey(tmp) < rskey(*(j-1)); --j) \
                    *j = *(j - 1); \
                *j = tmp; \
            } \
    } \
    \
    __device__ void rs_insertsort_d_##name(rstype_t *beg, rstype_t *end) \
    { \
        rstype_t *i; \
        for (i = beg + 1; i < end; ++i) \
            if (rskey(*i) < rskey(*(i - 1))) { \
                rstype_t *j, tmp = *i; \
                for (j = i; j > beg && rskey(tmp) < rskey(*(j-1)); --j) \
                    *j = *(j - 1); \
                *j = tmp; \
            } \
    } \
    \
    void rs_sort_##name(rstype_t *beg, rstype_t *end, int n_bits, int s) \
    { \
        rstype_t *i; \
        int size = 1<<n_bits, m = size - 1; \
        rsbucket_##name##_t *k, b[1<<RS_MAX_BITS], *be = b + size; \
        assert(n_bits <= RS_MAX_BITS); \
        for (k = b; k != be; ++k) k->b = k->e = beg; \
        for (i = beg; i != end; ++i) ++b[(rskey(*i) >> s) & m].e; \
        for (k = b + 1; k != be; ++k) \
            k->e += (k-1)->e - beg, k->b = (k-1)->e; \
        for (k = b; k != be;) { \
            if (k->b != k->e) { \
                rsbucket_##name##_t *l; \
                if ((l = b + ((rskey(*k->b) >> s) & m)) != k) { \
                    rstype_t tmp = *k->b, swap; \
                    do { \
                        swap = tmp; tmp = *l->b; *l->b++ = swap; \
                        l = b + ((rskey(tmp) >> s) & m); \
                    } while (l != k); \
                    *k->b++ = tmp; \
                } else ++k->b; \
            } else ++k; \
        } \
        for (b->b = beg, k = b + 1; k != be; ++k) k->b = (k-1)->e; \
        if (s) { \
            s = s > n_bits? s - n_bits : 0; \
            for (k = b; k != be; ++k) \
                if (k->e - k->b > RS_MIN_SIZE) rs_sort_##name(k->b, k->e, n_bits, s); \
                else if (k->e - k->b > 1) rs_insertsort_##name(k->b, k->e); \
        } \
    } \
    \
    typedef struct { \
        rstype_t* beg; \
        rstype_t* end; \
        int s; \
    } rs_frame_##name##_t; \
    \
    __device__ void rs_sort_d_##name(rstype_t *beg, rstype_t *end, int n_bits, int s_initial) \
    { \
        const int size = 1 << n_bits; \
        int m = size - 1; \
        __shared__ rsbucket_##name##_t b[1 << RS_MAX_BITS]; \
        rsbucket_##name##_t *be = b + size; \
        __shared__ rs_frame_##name##_t stack_##name[128];\
        int sp_##name = 0; \
        stack_##name[sp_##name++] = (rs_frame_##name##_t){beg, end, s_initial}; \
        while (sp_##name > 0) { \
            rs_frame_##name##_t frame = stack_##name[--sp_##name]; \
            rstype_t *f_beg = frame.beg; \
            rstype_t *f_end = frame.end; \
            int s = frame.s; \
            for (rsbucket_##name##_t *k = b; k != be; ++k) \
                k->b = k->e = f_beg; \
            for (rstype_t *i = f_beg; i != f_end; ++i) \
                ++b[(rskey(*i) >> s) & m].e; \
            for (rsbucket_##name##_t *k = b + 1; k != be; ++k) \
                k->e += (k-1)->e - f_beg, k->b = (k-1)->e; \
            for (rsbucket_##name##_t *k = b; k != be;) { \
                if (k->b != k->e) { \
                    rsbucket_##name##_t *l; \
                    if ((l = b + ((rskey(*k->b) >> s) & m)) != k) { \
                        rstype_t tmp = *k->b, swap; \
                        do { \
                            swap = tmp; \
                            tmp = *l->b; \
                            *l->b++ = swap; \
                            l = b + ((rskey(tmp) >> s) & m); \
                        } while (l != k); \
                        *k->b++ = tmp; \
                    } else ++k->b; \
                } else ++k; \
            } \
            /* fix: separo l'assegnazione dall'inizializzazione del for */ \
            b->b = f_beg; \
            for (rsbucket_##name##_t *k = b + 1; k != be; ++k) \
                k->b = (k-1)->e; \
            if (s > 0) { \
                int next_s = s > n_bits ? s - n_bits : 0; \
                for (rsbucket_##name##_t *k = b; k != be; ++k) { \
                    int len = k->e - k->b; \
                    if (len > RS_MIN_SIZE) \
                        stack_##name[sp_##name++] = (rs_frame_##name##_t){k->b, k->e, next_s}; \
                    else if (len > 1) \
                        rs_insertsort_d_##name(k->b, k->e); \
                } \
            } \
        } \
    } \
    \
    void radix_sort_h_##name(rstype_t *beg, rstype_t *end) \
    { \
        if (end - beg <= RS_MIN_SIZE) rs_insertsort_##name(beg, end); \
        else rs_sort_##name(beg, end, RS_MAX_BITS, (sizeof_key - 1) * RS_MAX_BITS); \
    } \
    \
    __device__ void radix_sort_d_##name(rstype_t *beg, rstype_t *end) \
    { \
        if (end - beg <= RS_MIN_SIZE) rs_insertsort_d_##name(beg, end); \
        else rs_sort_d_##name(beg, end, RS_MAX_BITS, (sizeof_key - 1) * RS_MAX_BITS); \
     }
#endif
