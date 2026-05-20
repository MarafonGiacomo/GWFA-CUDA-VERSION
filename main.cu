#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <zlib.h>

#include <time.h>
#ifdef GWF_ENABLE_DEBUG_LOG
#include <unistd.h>
#endif
#include <cuda_runtime.h>

#include "gfa.h"
#include "gfa-priv.h"
#include "gwfa.h"
#include "ketopt.h"
#include "ksort.h"

#include "kseq.h"

#define THREADS_PER_BLOCK 32
#define N_STREAMS 16

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, \
              __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

KSEQ_INIT(gzFile, gzread)

static double wall_time_seconds(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}


#ifdef GWF_ENABLE_DEBUG_LOG
#define GWF_PROGRESS_SLOTS 16

extern __device__ volatile unsigned long long *gwf_debug_progress;
extern __device__ int gwf_debug_enabled;
extern __device__ unsigned long long gwf_debug_stride;

static int env_int_default(const char *name, int default_value)
{
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : default_value;
}

static unsigned long long env_ull_default(const char *name, unsigned long long default_value)
{
    const char *v = getenv(name);
    return (v && *v) ? strtoull(v, NULL, 10) : default_value;
}

static const char *gwf_stage_name(unsigned long long stage)
{
    switch (stage) {
        case 1:  return "kernel-start";
        case 10: return "gwf-start";
        case 20: return "before-extend";
        case 30: return "extend-enter";
        case 40: return "extend-A-loop";
        case 50: return "copy-B-to-a";
        case 60: return "dedup-done";
        case 70: return "iteration-done";
        case 77: return "hash-probing";
        case 90: return "gwf-done";
        case 99: return "kernel-done";
        default: return "unknown";
    }
}

static void gwf_debug_print_progress(const volatile unsigned long long *p, int stream_id, unsigned long launch_id, size_t read_begin, size_t read_end, const char *prefix){
    
    unsigned long long block = p[0];
    unsigned long long stage = p[1];
    unsigned long long score_s = p[2];
    unsigned long long n_a = p[3];
    unsigned long long n_A = p[4];
    unsigned long long n_B = p[5];
    unsigned long long n_tmp = p[6];
    unsigned long long beat = p[8];
    unsigned long long warn = p[10];
    unsigned long long end_v = p[11];
    unsigned long long end_off = p[12];
    unsigned long long ql = p[13];

    fprintf(stderr, 
            "%s [stream %d launch %lu reads [%lu,%lu)) "
            "block=%llu stage=%llu(%s) s=%llu ql=%llu n_a=%llu n_A=%llu n_B=%llu n_tmp=%llu beat=%llu warn=%llu end=(%llu,%llu)\n",
            prefix,
            stream_id,
            launch_id,
            (unsigned long)read_begin,
            (unsigned long)read_end,
            block,
            stage,
            gwf_stage_name(stage),
            score_s,
            ql,
            n_a,
            n_A,
            n_B,
            n_tmp,
            beat,
            warn,
            end_v,
            end_off);
    fflush(stderr);
}

static void gwf_debug_poll_stream(cudaStream_t stream, int stream_id, unsigned long launch_id, size_t read_begin, size_t read_end, const volatile unsigned long long *progress_h, int poll_ms)
{
    double last_log = 0.0;
    if (poll_ms < 10) poll_ms = 10;

    for (;;) {
        cudaError_t q = cudaStreamQuery(stream);
        if (q == cudaSuccess) {
            gwf_debug_print_progress(progress_h, stream_id, launch_id, read_begin, read_end, "[GWFDBG done]");
            return;
        }
        if (q != cudaErrorNotReady) {
            fprintf(stderr, "CUDA stream error while polling stream %d launch %lu: %s\n",
                    stream_id, launch_id, cudaGetErrorString(q));
            exit(EXIT_FAILURE);
        }

        double now = wall_time_seconds();
        if (now - last_log >= 1.0) {
            gwf_debug_print_progress(progress_h, stream_id, launch_id, read_begin, read_end, "[GWFDBG wait]");
            last_log = now;
        }
        usleep((useconds_t)poll_ms * 1000u);
    }
}
#endif

__launch_bounds__(THREADS_PER_BLOCK, 1) __global__ void gwf_ed_kernel(char* const __restrict__ array_grande, const size_t* const __restrict__ index_array_d, const gwf_edbuf_t* const __restrict__ buf_reads_d, const char* __restrict__ all_sequences_d, const size_t* __restrict__ seq_offset_d, const size_t* __restrict__ seq_len_d, const gwf_graph_t* const __restrict__ g_d, int32_t* const __restrict__ scores_d, const int total_reads) 
{

	__shared__ uint64_t diag_shared[64];

    const int read_idx = blockIdx.x; 

	gwf_edbuf_t local_buf = buf_reads_d[read_idx];
	const char* __restrict__ seq_ptr = all_sequences_d + seq_offset_d[read_idx]; 
	const int ql = (int)seq_len_d[read_idx]; 

#ifdef GWF_ENABLE_DEBUG_LOG
	if (gwf_debug_enabled && gwf_debug_progress != NULL && threadIdx.x == 0) {
		volatile unsigned long long *p = gwf_debug_progress;
		p[0] = (unsigned long long)blockIdx.x;
		p[1] = 1ULL;
		p[2] = 0ULL;
		p[3] = 0ULL;
		p[4] = 0ULL;
		p[5] = 0ULL;
		p[6] = 0ULL;
		p[7] = (unsigned long long)clock64();
		p[8] = p[8] + 1ULL;
		p[13] = (unsigned long long)ql;
		__threadfence_system();
	}
#endif

	const char* __restrict__ ptr_base = array_grande + index_array_d[read_idx]; 
	const char* __restrict__ current_offset = ptr_base;

	local_buf.a_buf.vd = (uint64_t*)current_offset;
    current_offset += local_buf.max_a * sizeof(uint64_t);
    local_buf.a_buf.k = (int32_t*)current_offset;
    current_offset += local_buf.max_a * sizeof(int32_t);
    local_buf.a_buf.xo = (uint32_t*)current_offset;
    current_offset += local_buf.max_a * sizeof(uint32_t);

	local_buf.A = (gwf_diag_t*)current_offset;
    current_offset += local_buf.max_A * sizeof(gwf_diag_t);

	local_buf.B = (gwf_diag_t*)current_offset;
    current_offset += local_buf.max_B * sizeof(gwf_diag_t);
	

	local_buf.ooo = (gwf_diag_t*)current_offset;
    current_offset += local_buf.max_ooo * sizeof(gwf_diag_t);

	local_buf.intv.vd0 = (uint64_t*)current_offset;
    current_offset += local_buf.max_intv * sizeof(uint64_t);
    local_buf.intv.vd1 = (uint64_t*)current_offset;
    current_offset += local_buf.max_intv * sizeof(uint64_t);
    
    local_buf.tmp.vd0 = (uint64_t*)current_offset;
    current_offset += local_buf.max_tmp * sizeof(uint64_t);
    local_buf.tmp.vd1 = (uint64_t*)current_offset;
    current_offset += local_buf.max_tmp * sizeof(uint64_t);

    local_buf.swap.vd0 = (uint64_t*)current_offset;
    current_offset += local_buf.max_swap * sizeof(uint64_t);
    local_buf.swap.vd1 = (uint64_t*)current_offset;
    current_offset += local_buf.max_swap * sizeof(uint64_t);

	local_buf.ha = (uint64_t*)current_offset;
	

	int32_t final_score = gwf_ed(g_d, ql, seq_ptr, 0, -1, 0, &local_buf, diag_shared);

	scores_d[read_idx] = final_score;

#ifdef GWF_ENABLE_DEBUG_LOG
	if (gwf_debug_enabled && gwf_debug_progress != NULL && threadIdx.x == 0) {
		volatile unsigned long long *p = gwf_debug_progress;
		p[0] = (unsigned long long)blockIdx.x;
		p[1] = 99ULL;
		p[2] = (unsigned long long)(long long)final_score;
		p[7] = (unsigned long long)clock64();
		p[8] = p[8] + 1ULL;
		p[11] = (unsigned long long)(long long)final_score;
		p[15] = (unsigned long long)blockIdx.x;
		__threadfence_system();
	}
#endif
	

}

/**********************
 * Indexing the graph *
 **********************/

#define arc_key(x) ((x).a)
KRADIX_SORT_INIT_D(gwf_arc, gwf_arc_t, arc_key, 8)

// index the graph such that we can quickly access the neighbors of a vertex
void gwf_ed_index_arc_core(uint64_t *idx, uint32_t n_vtx, uint32_t n_arc, gwf_arc_t *arc)
{
	uint32_t i, st;
	radix_sort_h_gwf_arc(arc, arc + n_arc);

	for (st = 0, i = 1; i <= n_arc; ++i) {

		//si entra qui se siamo alla fine o alla fine del blocco
		if (i == n_arc || arc[i].a>>32 != arc[st].a>>32) {

			uint32_t v = arc[st].a>>32;
			assert(v < n_vtx);

			idx[v] = (uint64_t)st << 32 | (i - st);
			st = i;
		}
	}
}

void gwf_ed_index(gwf_graph_t *g)
{
	g->aux =(u_int64_t*)calloc(g->n_vtx,sizeof(*g->aux));
	
	gwf_ed_index_arc_core(g->aux, g->n_vtx, g->n_arc, g->arc);
}

// free the index
void gwf_cleanup(gwf_graph_t *g)
{
	free(g->aux);
	g->aux = 0;
}

inline uint32_t next_power_of_two(uint32_t x){
	if(x == 0){
		return x;
	}

	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;

	return x + 1;
}

gwf_graph_t *gwf_gfa2gwf(const gfa_t * __restrict__ gfa, uint32_t v0)
{
	int32_t i, k;
	gwf_graph_t *g;
	gfa_sub_t *sub;

	sub = gfa_sub_from(gfa, v0, 1<<30);

	GFA_CALLOC(g, 1);
	g->n_vtx = sub->n_v;
	g->n_arc = sub->n_a;
	GFA_MALLOC(g->len, g->n_vtx);
	GFA_MALLOC(g->src, g->n_vtx);
	GFA_MALLOC(g->seq, g->n_vtx);
	GFA_MALLOC(g->arc, g->n_arc);

	for (i = k = 0; i < sub->n_v; ++i) {

		uint32_t v = sub->v[i].v, len = gfa->seg[v>>1].len;
		uint32_t j;
		const gfa_seg_t *s = &gfa->seg[v>>1];

		g->len[i] = len;
		g->src[i] = v;
		GFA_MALLOC(g->seq[i], len + 1);

		if (v&1) {
            for (j = 0; j < len; ++j) {
                g->seq[i][j] = gfa_comp_table[(uint8_t)s->seq[len - j - 1]];
            }
        } else {
            memcpy(g->seq[i], s->seq, len);
        }

		g->seq[i][len] = 0; // null terminated for convenience

		int32_t l;
		for (l = 0; l < sub->v[i].n; ++l) {
			uint64_t a = sub->a[sub->v[i].off + l];

			g->arc[k].a = (uint64_t)i<<32 | a>>32;
			g->arc[k].o = gfa->arc[(uint32_t)a].ow;
			++k;
		}
		assert(k <= g->n_arc);
	}
	free(sub->a);
    free(sub->v); 
    free(sub);
	
	return g;
}

void gwf_free(gwf_graph_t *g)
{
	uint32_t i;
	for (i = 0; i < g->n_vtx; ++i) { free(g->seq[i]); }
	free(g->len); free(g->seq); free(g->arc); free(g->src); free(g);
}

void gwf_graph_print(FILE * __restrict__ fp, const gwf_graph_t * __restrict__ g)
{
	uint32_t i;
	for (i = 0; i < g->n_vtx; ++i) { fprintf(fp, "S\t%d\t%s\tLN:i:%d\n", i, g->seq[i], g->len[i]); }
	for (i = 0; i < g->n_arc; ++i) { fprintf(fp, "L\t%d\t+\t%d\t+\t%dM\n", (uint32_t)(g->arc[i].a>>32), (uint32_t)g->arc[i].a, g->arc[i].o); }
}

int main(int argc, char *argv[])
{
	gzFile fp;
	kseq_t *ks;
	ketopt_t o = KETOPT_INIT;
	gfa_t *gfa;
	gwf_graph_t *g;
	
	int c, print_graph = 0;	
	uint32_t v0 = 0<<1|0; // first segment, forward strand
	// uint32_t max_lag = 0;

	char *sname = 0;

	while ((c = ketopt(&o, argc, argv, 1, "ptl:s:", 0)) >= 0) {
		if (c == 'p') { print_graph = 1; }
		// else if (c == 'l') max_lag = atoi(o.arg);
		else if (c == 's') { sname = o.arg; }
		
	}
	if ((!print_graph && argc - o.ind < 2) || (print_graph && argc == o.ind)) {
		fprintf(stderr, "Usage: gwf-test [options] <target.gfa|fa> <query.fa>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -l INT    max lag behind the furthest wavefront; 0 to disable [0]\n");
		fprintf(stderr, "  -s STR    starting segment name [first]\n");
		
		fprintf(stderr, "  -p        output GFA in the forward strand\n");
		return 1;
	}

	cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
	if (flags_err != cudaSuccess && flags_err != cudaErrorSetOnActiveProcess) {
		fprintf(stderr, "WARNING: cudaSetDeviceFlags(cudaDeviceMapHost) failed: %s\n", cudaGetErrorString(flags_err));
	}

	gfa = gfa_read(argv[o.ind]);
	assert(gfa);

	if (sname) {
		int32_t sid;
		sid = gfa_name2id(gfa, sname);
		if (sid < 0) { 
            fprintf(stderr, "ERROR: failed to find segment '%s'\n", sname);
        } else { v0 = sid<<1 | 0; } // TODO: also allow to change the orientation
	}

	g = gwf_gfa2gwf(gfa, v0);

	if (print_graph) {
		gwf_graph_print(stdout, g);
		return 0; // free memory
	}


    gwf_ed_index(g);

	gwf_graph_t g_h = *g;       
    gwf_graph_t *g_d = NULL;     

    CHECK(cudaMalloc(&g_d, sizeof(gwf_graph_t)));

    CHECK(cudaMalloc(&g_h.len, g->n_vtx * sizeof(uint32_t)));
    CHECK(cudaMemcpy(g_h.len, g->len, g->n_vtx * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&g_h.src, g->n_vtx * sizeof(uint32_t)));
    CHECK(cudaMemcpy(g_h.src, g->src, g->n_vtx * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&g_h.arc, g->n_arc * sizeof(gwf_arc_t)));
    CHECK(cudaMemcpy(g_h.arc, g->arc, g->n_arc * sizeof(gwf_arc_t), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&g_h.aux, g->n_vtx * sizeof(uint64_t)));
    CHECK(cudaMemcpy(g_h.aux, g->aux, g->n_vtx * sizeof(uint64_t), cudaMemcpyHostToDevice));

    char **seq_d;
    CHECK(cudaMalloc(&seq_d, g->n_vtx * sizeof(char*)));

    char **seq_tmp = (char**)malloc(g->n_vtx * sizeof(char*));

    for (uint32_t i = 0; i < g->n_vtx; i++) {
        size_t len = strlen(g->seq[i]) + 1;
        char *s_d;
        CHECK(cudaMalloc(&s_d, len * sizeof(char)));
        CHECK(cudaMemcpy(s_d, g->seq[i], len * sizeof(char), cudaMemcpyHostToDevice));
        seq_tmp[i] = s_d; 
    }

    CHECK(cudaMemcpy(seq_d, seq_tmp, g->n_vtx * sizeof(char*), cudaMemcpyHostToDevice));
    free(seq_tmp);

    g_h.seq = seq_d; 

    CHECK(cudaMemcpy(g_d, &g_h, sizeof(gwf_graph_t), cudaMemcpyHostToDevice));

	size_t free_mem, total_mem;
	size_t GPU_POOL_SIZE;



    fp = gzopen(argv[o.ind+1], "r");
    assert(fp);
    ks = kseq_init(fp);
    int num_righe = 0;
    size_t total_seq_bytes = 0;

    while(kseq_read(ks) >= 0){
        num_righe++;
        total_seq_bytes += ks->seq.l + 1;
    }

    printf("\n righe: %d\n", num_righe);

    size_t* byte_read_singola = (size_t*) malloc(sizeof(size_t) * num_righe);

    gwf_edbuf_t* buf_reads = NULL;
    char* all_sequences_host = NULL;
    size_t* seq_len_host = NULL;
    size_t* seq_offset_host = (size_t*) malloc(sizeof(size_t) * num_righe);

    CHECK(cudaMallocHost((void**)&buf_reads, sizeof(gwf_edbuf_t) * (size_t)num_righe));
    CHECK(cudaMallocHost((void**)&seq_len_host, sizeof(size_t) * (size_t)num_righe));
    CHECK(cudaMallocHost((void**)&all_sequences_host, total_seq_bytes));

    int curr_read = 0;
    size_t total_bytes = 0;
    total_seq_bytes = 0;

    kseq_destroy(ks);
    gzclose(fp);

    fp = gzopen(argv[o.ind+1], "r");
    assert(fp);
    ks = kseq_init(fp);

    while(kseq_read(ks) >= 0){

        memset(&(buf_reads[curr_read]), 0, sizeof(gwf_edbuf_t)); 

        buf_reads[curr_read].max_a = (uint32_t) (((4 * ks->seq.l * 0.1) + 1) * 4);
        buf_reads[curr_read].max_B = 3 * buf_reads[curr_read].max_a + 1;
        buf_reads[curr_read].max_A = buf_reads[curr_read].max_a;
        buf_reads[curr_read].max_intv = buf_reads[curr_read].max_a;
        buf_reads[curr_read].max_tmp = buf_reads[curr_read].max_B;
        buf_reads[curr_read].max_swap = buf_reads[curr_read].max_a;
        buf_reads[curr_read].max_ooo = buf_reads[curr_read].max_B;
        buf_reads[curr_read].ha_mask = next_power_of_two(ks->seq.l * 32) - 1;

        size_t s = 0;
        s += buf_reads[curr_read].max_a * sizeof(uint64_t); // vd
        s += buf_reads[curr_read].max_a * sizeof(int32_t);  // k
        s += buf_reads[curr_read].max_a * sizeof(uint32_t); // xo
        s += buf_reads[curr_read].max_A * sizeof(gwf_diag_t);
        s += buf_reads[curr_read].max_B * sizeof(gwf_diag_t);
        s += buf_reads[curr_read].max_ooo * sizeof(gwf_diag_t);
        s += (buf_reads[curr_read].max_intv * 2) * sizeof(uint64_t); // vd0, vd1
        s += (buf_reads[curr_read].max_tmp * 2) * sizeof(uint64_t);  // vd0, vd1
        s += (buf_reads[curr_read].max_swap * 2) * sizeof(uint64_t); // vd0, vd1
        s += (buf_reads[curr_read].ha_mask + 1) * sizeof(uint64_t);

        byte_read_singola[curr_read] = s;
        total_bytes += s ;

        size_t seq_len = ks->seq.l;
        size_t seq_size_with_null = seq_len + 1;

        seq_len_host[curr_read] = seq_len;
        
        seq_offset_host[curr_read] = total_seq_bytes;
        
        memcpy(all_sequences_host + total_seq_bytes, ks->seq.s, seq_size_with_null);

        total_seq_bytes += seq_size_with_null;

        curr_read++; 
    }

    int32_t *scores_host = NULL;
    size_t num_reads = curr_read;
    CHECK(cudaMallocHost((void**)&scores_host, num_reads * sizeof(int32_t)));
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    GPU_POOL_SIZE = (free_mem * 90) / 100;

    int n_streams = N_STREAMS;
    if ((size_t)n_streams > num_reads) { n_streams = (int)num_reads; }
    if (n_streams < 1) { n_streams = 1; }


    size_t max_single_read_bytes = 0;
    for (size_t r = 0; r < num_reads; ++r) {
        size_t seq_bytes = seq_len_host[r] + 1;
        size_t off = 0;
        off = (off + byte_read_singola[r] + 63) & ~(size_t)63;
        off = (off + seq_bytes + 63) & ~(size_t)63;
        off = (off + sizeof(gwf_edbuf_t) + 63) & ~(size_t)63;
        off = (off + sizeof(size_t) + 63) & ~(size_t)63; // index_array
        off = (off + sizeof(size_t) + 63) & ~(size_t)63; // seq_offset
        off = (off + sizeof(size_t) + 63) & ~(size_t)63; // seq_len
        off = (off + sizeof(int32_t) + 63) & ~(size_t)63; // scores
        if (off > max_single_read_bytes) { max_single_read_bytes = off; }
    }

    while (n_streams > 1 && (GPU_POOL_SIZE / (size_t)n_streams) < max_single_read_bytes) {
        --n_streams;
    }

    size_t stream_capacity = GPU_POOL_SIZE / (size_t)n_streams;
    if (stream_capacity < max_single_read_bytes) {
        fprintf(stderr, "ERRORE: una singola read richiede %lu bytes, ma lo stream ha solo %lu bytes disponibili.\n", (unsigned long)max_single_read_bytes, (unsigned long)stream_capacity);
        exit(EXIT_FAILURE);
    }

    printf("\nGPU pool: %lu bytes | stream: %d | bytes/stream: %lu\n", (unsigned long)GPU_POOL_SIZE, n_streams, (unsigned long)stream_capacity);

    cudaStream_t* streams = (cudaStream_t*)malloc(n_streams * sizeof(cudaStream_t));
    char** arena_d = (char**)malloc(n_streams * sizeof(char*));
    size_t** index_stage_h = (size_t**)malloc(n_streams * sizeof(size_t*));
    size_t** seq_offset_stage_h = (size_t**)malloc(n_streams * sizeof(size_t*));
    size_t* stage_capacity = (size_t*)calloc(n_streams, sizeof(size_t));
    int* busy = (int*)calloc(n_streams, sizeof(int));

    assert(streams && arena_d && index_stage_h && seq_offset_stage_h && stage_capacity && busy);

    for (int s = 0; s < n_streams; ++s) {
        CHECK(cudaStreamCreate(&streams[s]));
        CHECK(cudaMalloc((void**)&arena_d[s], stream_capacity));
        index_stage_h[s] = NULL;
        seq_offset_stage_h[s] = NULL;
    }

#ifdef GWF_ENABLE_DEBUG_LOG
    int debug_progress_enabled = env_int_default("GWF_DEBUG", 1);
    int debug_poll_ms = env_int_default("GWF_DEBUG_POLL_MS", 250);
    unsigned long long debug_stride = env_ull_default("GWF_DEBUG_STRIDE", 100ULL);
    volatile unsigned long long *debug_progress_h = NULL;
    unsigned long long *debug_progress_d = NULL;

    if (debug_progress_enabled) {

        CHECK(cudaHostAlloc((void**)&debug_progress_h, GWF_PROGRESS_SLOTS * sizeof(unsigned long long), cudaHostAllocMapped));

        memset((void*)debug_progress_h, 0, GWF_PROGRESS_SLOTS * sizeof(unsigned long long));

        CHECK(cudaHostGetDevicePointer((void**)&debug_progress_d, (void*)debug_progress_h, 0));

        CHECK(cudaMemcpyToSymbol(gwf_debug_progress, &debug_progress_d, sizeof(debug_progress_d)));
        CHECK(cudaMemcpyToSymbol(gwf_debug_enabled, &debug_progress_enabled, sizeof(debug_progress_enabled)));
        CHECK(cudaMemcpyToSymbol(gwf_debug_stride, &debug_stride, sizeof(debug_stride)));
        fprintf(stderr, "[GWFDBG] enabled | poll=%d ms | stride=%llu score-iterations | set GWF_DEBUG=0 to disable\n", debug_poll_ms, debug_stride);
        fflush(stderr);
    } else {
        CHECK(cudaMemcpyToSymbol(gwf_debug_enabled, &debug_progress_enabled, sizeof(debug_progress_enabled)));
    }
#endif

    double t0 = wall_time_seconds();

    size_t next_read = 0;
    size_t launch_id = 0;

    while (next_read < num_reads) {
        int s = (int)(launch_id % (size_t)n_streams);

        if (busy[s]) {

            fprintf(stderr, "[STREAM %d] wait before reuse | launch %lu\n", s, (unsigned long)launch_id);
            fflush(stderr);
            CHECK(cudaStreamSynchronize(streams[s]));
            busy[s] = 0;
        }

        size_t seq_base = seq_offset_host[next_read];
        size_t work_bytes = 0;
        size_t seq_bytes = 0;
        int n_reads_stream = 0;

        while (next_read + (size_t)n_reads_stream < num_reads) {

            size_t r = next_read + (size_t)n_reads_stream;

            size_t read_work_start = (work_bytes + 63) & ~(size_t)63;
            size_t candidate_work_bytes = read_work_start + byte_read_singola[r];
            size_t candidate_seq_bytes = (seq_offset_host[r] + seq_len_host[r] + 1) - seq_base;
            int candidate_n_reads = n_reads_stream + 1;

            size_t off = 0;

            off = (off + candidate_work_bytes + 63) & ~(size_t)63;
            off = (off + candidate_seq_bytes + 63) & ~(size_t)63;
            off = (off + (size_t)candidate_n_reads * sizeof(gwf_edbuf_t) + 63) & ~(size_t)63;
            off = (off + (size_t)candidate_n_reads * sizeof(size_t) + 63) & ~(size_t)63; // index_array
            off = (off + (size_t)candidate_n_reads * sizeof(size_t) + 63) & ~(size_t)63; // seq_offset
            off = (off + (size_t)candidate_n_reads * sizeof(size_t) + 63) & ~(size_t)63; // seq_len
            off = (off + (size_t)candidate_n_reads * sizeof(int32_t) + 63) & ~(size_t)63; // scores

            if (off > stream_capacity) {

                if (n_reads_stream == 0) {
                    fprintf(stderr, "ERRORE: la read %lu non entra nello stream (%lu bytes richiesti, %lu disponibili).\n", (unsigned long)r, (unsigned long)off, (unsigned long)stream_capacity);
                    exit(EXIT_FAILURE);
                }
                break;
            }

            work_bytes = candidate_work_bytes;
            seq_bytes = candidate_seq_bytes;
            n_reads_stream = candidate_n_reads;
        }

        if ((size_t)n_reads_stream > stage_capacity[s]) {
            if (index_stage_h[s]) { CHECK(cudaFreeHost(index_stage_h[s])); }
            if (seq_offset_stage_h[s]) { CHECK(cudaFreeHost(seq_offset_stage_h[s])); }
            CHECK(cudaMallocHost((void**)&index_stage_h[s], (size_t)n_reads_stream * sizeof(size_t)));
            CHECK(cudaMallocHost((void**)&seq_offset_stage_h[s], (size_t)n_reads_stream * sizeof(size_t)));
            stage_capacity[s] = (size_t)n_reads_stream;
        }

        size_t tmp_work_bytes = 0;
        for (int j = 0; j < n_reads_stream; ++j) {
            size_t r = next_read + (size_t)j;
            size_t read_work_start = (tmp_work_bytes + 63) & ~(size_t)63;
            index_stage_h[s][j] = read_work_start;
            seq_offset_stage_h[s][j] = seq_offset_host[r] - seq_base;
            tmp_work_bytes = read_work_start + byte_read_singola[r];
        }

        size_t off = 0;
        char* array_d = arena_d[s] + off;
        off = (off + work_bytes + 63) & ~(size_t)63;

        char* sequences_d = arena_d[s] + off;
        off = (off + seq_bytes + 63) & ~(size_t)63;

        gwf_edbuf_t* buf_reads_d = (gwf_edbuf_t*)(arena_d[s] + off);
        off = (off + (size_t)n_reads_stream * sizeof(gwf_edbuf_t) + 63) & ~(size_t)63;

        size_t* index_array_d = (size_t*)(arena_d[s] + off);
        off = (off + (size_t)n_reads_stream * sizeof(size_t) + 63) & ~(size_t)63;

        size_t* seq_offset_d = (size_t*)(arena_d[s] + off);
        off = (off + (size_t)n_reads_stream * sizeof(size_t) + 63) & ~(size_t)63;

        size_t* seq_len_d = (size_t*)(arena_d[s] + off);
        off = (off + (size_t)n_reads_stream * sizeof(size_t) + 63) & ~(size_t)63;

        int32_t* scores_d = (int32_t*)(arena_d[s] + off);

        CHECK(cudaMemsetAsync(array_d, 0, work_bytes, streams[s]));
        
        CHECK(cudaMemcpyAsync(buf_reads_d, buf_reads + next_read, (size_t)n_reads_stream * sizeof(gwf_edbuf_t), cudaMemcpyHostToDevice, streams[s]));
        CHECK(cudaMemcpyAsync(index_array_d, index_stage_h[s], (size_t)n_reads_stream * sizeof(size_t), cudaMemcpyHostToDevice, streams[s]));
        CHECK(cudaMemcpyAsync(seq_offset_d, seq_offset_stage_h[s], (size_t)n_reads_stream * sizeof(size_t), cudaMemcpyHostToDevice, streams[s]));
        CHECK(cudaMemcpyAsync(seq_len_d, seq_len_host + next_read, (size_t)n_reads_stream * sizeof(size_t), cudaMemcpyHostToDevice, streams[s]));
        CHECK(cudaMemcpyAsync(sequences_d, all_sequences_host + seq_base, seq_bytes, cudaMemcpyHostToDevice, streams[s]));

        fprintf(stderr, "[STREAM %d] launch %lu | reads [%lu, %lu) | n=%d | work=%lu B | seq=%lu B | arena_used=%lu/%lu B\n", s, (unsigned long)launch_id, (unsigned long)next_read, (unsigned long)(next_read + (size_t)n_reads_stream), n_reads_stream, (unsigned long)work_bytes, (unsigned long)seq_bytes, (unsigned long)(off + (size_t)n_reads_stream * sizeof(int32_t)), (unsigned long)stream_capacity);
        fflush(stderr);

        gwf_ed_kernel<<<n_reads_stream, THREADS_PER_BLOCK, 0, streams[s]>>>(array_d, index_array_d, buf_reads_d, sequences_d, seq_offset_d, seq_len_d, g_d, scores_d, n_reads_stream);
        CHECK(cudaPeekAtLastError());

        CHECK(cudaMemcpyAsync(scores_host + next_read, scores_d, (size_t)n_reads_stream * sizeof(int32_t), cudaMemcpyDeviceToHost, streams[s]));

#ifdef GWF_ENABLE_DEBUG_LOG
        if (debug_progress_enabled && debug_progress_h != NULL) {
            gwf_debug_poll_stream(streams[s], s, (unsigned long)launch_id, next_read, next_read + (size_t)n_reads_stream, debug_progress_h, debug_poll_ms);
            busy[s] = 0;
        } else
#endif
        {
            busy[s] = 1;
        }

        next_read += (size_t)n_reads_stream;
        ++launch_id;
    }

    for (int s = 0; s < n_streams; ++s) {
        if (busy[s]) {
            fprintf(stderr, "[STREAM %d] final synchronize\n", s);
            fflush(stderr);
            CHECK(cudaStreamSynchronize(streams[s]));
        }
    }

    double elapsed = wall_time_seconds() - t0;

    for (size_t j = 0; j < num_reads; ++j) {
        printf("%d\n", scores_host[j]);
    }
    printf("\nTempo GPU streams: %.6f secondi\n", elapsed);

    CHECK(cudaFreeHost(buf_reads));
    CHECK(cudaFreeHost(seq_len_host));
    CHECK(cudaFreeHost(all_sequences_host));
    CHECK(cudaFreeHost(scores_host));
#ifdef GWF_ENABLE_DEBUG_LOG
    if (debug_progress_h != NULL) { CHECK(cudaFreeHost((void*)debug_progress_h)); }
#endif

    for (int s = 0; s < n_streams; ++s) {
        if (index_stage_h[s]) { CHECK(cudaFreeHost(index_stage_h[s])); }
        if (seq_offset_stage_h[s]) { CHECK(cudaFreeHost(seq_offset_stage_h[s])); }
        CHECK(cudaFree(arena_d[s]));
        CHECK(cudaStreamDestroy(streams[s]));
    }

    free(streams);
    free(arena_d);
    free(index_stage_h);
    free(seq_offset_stage_h);
    free(stage_capacity);
    free(busy);

    free(byte_read_singola);
    free(seq_offset_host);

    CHECK(cudaFree(g_h.len));
    CHECK(cudaFree(g_h.src));
    CHECK(cudaFree(g_h.arc));
    CHECK(cudaFree(g_h.aux));
    CHECK(cudaFree(g_h.seq));
    CHECK(cudaFree(g_d));

	
	
	kseq_destroy(ks);
	gzclose(fp);
	gfa_destroy(gfa);

	gwf_cleanup(g);
	gwf_free(g);
	
	return 0;
}

