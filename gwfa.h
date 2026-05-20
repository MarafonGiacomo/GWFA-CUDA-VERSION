#ifndef GWFA_H
#define GWFA_H

#include <stdint.h>

typedef struct {
	uint64_t a;
	int32_t o;
} gwf_arc_t;

typedef struct {
	uint32_t n_vtx, n_arc;
	uint32_t *len;
	uint32_t *src;
	char **seq;
	gwf_arc_t *arc;
	uint64_t *aux;
} gwf_graph_t;

typedef struct{ // a diagonal
	uint64_t* __restrict__ vd; // higher 32 bits: vertex ID; lower 32 bits: diagonal+0x4000000
	int32_t* __restrict__ k;
	uint32_t* __restrict__ xo; // higher 31 bits: anti diagonal; lower 1 bit: out-of-order or not
	
} gwf_diag_soa_t;

typedef struct {
	uint64_t vd; // higher 32 bits: vertex ID; lower 32 bits: diagonal+0x4000000
	int32_t k;
	uint32_t xo;
} gwf_diag_t;

typedef struct {
	uint64_t vd0; 
	uint64_t vd1;
} gwf_intv_t;

typedef struct __align__(16){
	uint64_t* __restrict__ vd0; 
	uint64_t* __restrict__ vd1;
} gwf_intv_soa_t;


typedef struct {
	int32_t n_intv;
	int32_t n_tmp;
	int32_t n_swap;
	int32_t n_ooo;
	int32_t n_A;
	int32_t n_B;
	int32_t n_a;
} counters;

typedef struct{
	gwf_intv_soa_t intv;
	int32_t max_intv;

	gwf_intv_soa_t tmp;
	int32_t max_tmp;

	gwf_intv_soa_t swap;
	int32_t max_swap;

	gwf_diag_t* ooo;
	int32_t max_ooo;

	gwf_diag_soa_t a_buf;	
	int32_t max_a;

	gwf_diag_t* B;
	int32_t max_B;

	gwf_diag_t* __restrict__ A;
	int32_t max_A;
	int32_t head_A;
	int32_t tail_A;

	u_int64_t* __restrict__ ha;
	uint32_t ha_mask;
} gwf_edbuf_t;


#ifdef __cplusplus
extern "C" {
#endif

void gwf_ed_index(gwf_graph_t *g);
void gwf_cleanup(gwf_graph_t *g);


#ifdef __cplusplus
}
#endif

	__device__ int32_t gwf_ed(const gwf_graph_t *g, int32_t ql, const char *q, int32_t v0, int32_t v1, uint32_t max_lag, gwf_edbuf_t *buf, uint64_t *diag_shared);

// __global__ void gwf_ed_kernel(const char* const __restrict__ array_grande, const size_t* const __restrict__ index_array_d, const gwf_edbuf_t* const __restrict__ buf_reads_d, const char* __restrict__ all_sequences_d, const size_t* __restrict__ seq_offset_d, const size_t* __restrict__ seq_len_d, const gwf_graph_t* const __restrict__ g_d, int32_t* const __restrict__ scores_d, const int total_reads);

#endif
