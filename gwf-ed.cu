
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "gwfa.h"

/**************************************
 * Graph WaveFront with edit distance *
 **************************************/

#define GWF_DIAG_SHIFT 0x40000000

#define THREADS_PER_BLOCK 32


#ifdef GWF_ENABLE_DEBUG_LOG
#define GWF_DBG_STAGE_GWF_START       10
#define GWF_DBG_STAGE_BEFORE_EXTEND   20
#define GWF_DBG_STAGE_EXTEND_ENTER    30
#define GWF_DBG_STAGE_EXTEND_A_LOOP   40
#define GWF_DBG_STAGE_COPY_B_TO_A     50
#define GWF_DBG_STAGE_DEDUP_DONE      60
#define GWF_DBG_STAGE_ITERATION_DONE  70
#define GWF_DBG_STAGE_HASH_PROBING    77
#define GWF_DBG_STAGE_GWF_DONE        90

__device__ volatile unsigned long long *gwf_debug_progress = NULL;
__device__ int gwf_debug_enabled = 0;
__device__ unsigned long long gwf_debug_stride = 100ULL;

__device__ __forceinline__ static void gwf_debug_mark(int stage,
                                                       int32_t score_s,
                                                       int32_t n_a,
                                                       int32_t n_A,
                                                       int32_t n_B,
                                                       int32_t n_tmp,
                                                       int32_t end_v,
                                                       int32_t end_off,
                                                       int32_t ql)
{
	if (!gwf_debug_enabled || gwf_debug_progress == NULL || threadIdx.x != 0)
		return;

	volatile unsigned long long *p = gwf_debug_progress;
	p[0] = (unsigned long long)blockIdx.x;
	p[1] = (unsigned long long)stage;
	p[2] = (unsigned long long)(long long)score_s;
	p[3] = (unsigned long long)(long long)n_a;
	p[4] = (unsigned long long)(long long)n_A;
	p[5] = (unsigned long long)(long long)n_B;
	p[6] = (unsigned long long)(long long)n_tmp;
	p[7] = (unsigned long long)clock64();
	p[8] = p[8] + 1ULL;
	if ((unsigned long long)(long long)score_s > p[9])
		p[9] = (unsigned long long)(long long)score_s;
	p[11] = (unsigned long long)(long long)end_v;
	p[12] = (unsigned long long)(long long)end_off;
	p[13] = (unsigned long long)(long long)ql;
	__threadfence_system();
}

__device__ __forceinline__ static void gwf_debug_hash_probe_mark(int32_t probes)
{
	if (!gwf_debug_enabled || gwf_debug_progress == NULL || threadIdx.x != 0)
		return;

	volatile unsigned long long *p = gwf_debug_progress;
	p[0] = (unsigned long long)blockIdx.x;
	p[1] = (unsigned long long)GWF_DBG_STAGE_HASH_PROBING;
	p[7] = (unsigned long long)clock64();
	p[8] = p[8] + 1ULL;
	p[10] = (unsigned long long)(long long)probes;
	__threadfence_system();
}
#else
#define gwf_debug_mark(...) do { } while (0)
#define gwf_debug_hash_probe_mark(...) do { } while (0)
#endif

__device__ __inline__ static uint64_t gwf_gen_vd(uint32_t v, int32_t d)
{
	return (uint64_t)v << 32 | (GWF_DIAG_SHIFT + d);
}

__device__ __inline__ void fast_parallel_sort_odd_even(uint64_t *keys, uint64_t *vals, int n)
{
	const int32_t tid = threadIdx.x;
	const int32_t bdim = blockDim.x;

	for (int32_t i = 0; i < n; i++)
	{
		int32_t offset = i & 1;

		for (int32_t j_base = 0; j_base < n - 1; j_base += bdim * 2)
		{
			int32_t j = j_base + offset + (tid * 2);

			if (j < n - 1)
			{
				uint64_t k1 = keys[j];
				uint64_t k2 = keys[j + 1];

				if (k1 > k2)
				{
					keys[j] = k2;
					keys[j + 1] = k1;

					uint64_t v1 = vals[j];
					uint64_t v2 = vals[j + 1];
					vals[j] = v2;
					vals[j + 1] = v1;
				}
			}
		}
		__syncthreads();
	}
}

__device__ __inline__ static void fast_parallel_sort_merge(uint64_t *__restrict__ keys, uint64_t *__restrict__ vals, int32_t n, uint64_t *__restrict__ tmp_keys, uint64_t *__restrict__ tmp_vals){
	uint64_t *src_k = keys;
	uint64_t *src_v = vals;
	uint64_t *dst_k = tmp_keys;
	uint64_t *dst_v = tmp_vals;

	for (int32_t width = 1; width < n; width <<= 1)
	{
		for (int32_t left = 0; left < n; left += (width << 1))
		{
			int32_t mid = left + width;
			int32_t right = left + (width << 1);

			if (mid > n)
				mid = n;
			if (right > n)
				right = n;

			const int32_t n1 = mid - left;
			const int32_t n2 = right - mid;
			const int32_t total = n1 + n2;

			for (int32_t out = threadIdx.x; out < total; out += blockDim.x)
			{
				int32_t low = max((int32_t)0, out - n2);
				int32_t high = min(out, n1);

				while (low < high)
				{
					int32_t m = (low + high) >> 1;
					int32_t j = out - 1 - m;

					bool cond = src_k[left + m] <= src_k[mid + j];

					low = cond ? m + 1 : low;
					high = cond ? high : m;
				}

				int32_t i = low;
				int32_t j = out - i;
				bool take_left = i < n1 && (j >= n2 || src_k[left + i] <= src_k[mid + j]);

				if (take_left)
				{
					dst_k[left + out] = src_k[left + i];
					dst_v[left + out] = src_v[left + i];
				}
				else
				{
					dst_k[left + out] = src_k[mid + j];
					dst_v[left + out] = src_v[mid + j];
				}
			}
		}

		__syncthreads();

		uint64_t *tk = src_k;
		src_k = dst_k;
		dst_k = tk;

		uint64_t *tv = src_v;
		src_v = dst_v;
		dst_v = tv;

		// __syncthreads();

	}

	if (src_k != keys)
	{
		for (int32_t i = threadIdx.x; i < n; i += blockDim.x)
		{
			keys[i] = src_k[i];
			vals[i] = src_v[i];
		}

		__syncthreads();
	}
}

__device__ __inline__ void fast_parallel_sort(uint64_t *__restrict__ keys, uint64_t *__restrict__ vals, int32_t n, uint64_t *__restrict__ tmp_keys, uint64_t *__restrict__ tmp_vals, int32_t max_tmp){
	if (n <= 1)
		return;

	if (n > 64 && n <= max_tmp)
	{
		fast_parallel_sort_merge(keys, vals, n, tmp_keys, tmp_vals);
		return;
	}

	fast_parallel_sort_odd_even(keys, vals, n);
}

__device__ __inline__ void fast_parallel_sort(uint64_t *keys, uint64_t *vals, int32_t n)
{
	fast_parallel_sort_odd_even(keys, vals, n);
}

// this function checks if the vertexes are orderd correctly. If not, then returns the index of the vertex that is in the wrong order
__device__ __inline__ static int gwf_intv_is_sorted(int32_t n_a, const gwf_intv_soa_t a, uint64_t *diag_shared)
{
	int32_t idx = 0;
	int32_t curr_buf = 0;
	int32_t next_buf = 32;

	if (idx + threadIdx.x < n_a)
	{
		diag_shared[curr_buf + threadIdx.x] = a.vd0[idx + threadIdx.x];
	}
	idx = 1;
	uint64_t first_vd = diag_shared[0];
	uint64_t mask = 0;
	idx += 32;
	while (idx - 32 < n_a)
	{
		bool in_range_next = idx + threadIdx.x < n_a;
		if (in_range_next)
		{
			diag_shared[next_buf + threadIdx.x] = a.vd0[idx + threadIdx.x];
		}
		bool in_range_curr = (idx - 32) + threadIdx.x < n_a;
		uint64_t curr = in_range_curr ? diag_shared[curr_buf + threadIdx.x] : 0;
		// takes the value of thread k - 1.
		uint64_t prev = __shfl_up_sync(0xffffffff, curr, 1);

		prev = (threadIdx.x == 0) ? first_vd : prev;

		mask = __ballot_sync(0xffffffff, in_range_curr && prev > curr);
		if (mask != 0)
		{
			return 0;
		}

		// everyone reads the value of thread 31 (this is the one needed for thread 0 on the next iteration)
		first_vd = __shfl_sync(0xffffffff, curr, 31);

		idx += blockDim.x;

		int32_t tmp = curr_buf;
		curr_buf = next_buf;
		next_buf = tmp;
	}
	return 1;
}

// merge overlapping intervals; input must be sorted
__device__ __inline__ static int32_t gwf_intv_merge_adj(int32_t n, gwf_intv_soa_t a)
{

	// a => intv (the new one, generated from the merge of tmp and swap (== old intv))
	int32_t i, k;
	// start and end, this is for managing the intervals
	uint64_t st, en;
	if (n == 0)
		return 0;
	// loads the intervals for the first "state" of intv
	st = a.vd0[0], en = a.vd1[0];
	// merging overlapping intervals
	for (i = 1, k = 0; i < n; ++i)
	{
		// this means that there's no more overlapping / close together diags
		if (a.vd0[i] > en)
		{
			// lock the interval in the first and second element of a. Meaning that, if 10 states of a were close, they will be just deleted and consideres as in interval in a[0].vd0 and a[0].vd1.
			a.vd0[k] = st, a.vd1[k++] = en;
			// sets the new start and end
			st = a.vd0[i], en = a.vd1[i];
			// this choses the furthest one to be the end.
		}
		else
			en = en > a.vd1[i] ? en : a.vd1[i];
	}
	// exact same as the concept above
	a.vd0[k] = st, a.vd1[k++] = en;
	// k will be the number of elements in the new swap array. Also it states the number of different intervals
	return k;
}


// merge two sorted interval lists. Used for intv (old memory) and swap (new memory, current tmp / B buffer) and tmp
__device__ __inline__ static int32_t gwf_intv_merge2(gwf_intv_soa_t const a, const int32_t n_b, const gwf_intv_soa_t b, int32_t n_c, const gwf_intv_soa_t c)
{
	// a => intv
	// b => swap (still intv but a copy)
	// c => tmp (current new states)

	// total array size
	const int32_t total = n_b + n_c;
	if (total == 0)
		return 0;
	int32_t base_k = 0;

	// merge path algorithm. k is the index in wich a certain element should appear in the final array
	while (base_k < total)
	{
		int32_t k = base_k + threadIdx.x;
		bool in_range = k < total;

		if (in_range)
		{
			// this is the minimum elements that you'll need to take from b. If k = 100 and n_c is small (20) then you'll need 80 elements from b
			int32_t low = max((int32_t)0, (int32_t)(k - n_c));
			// this is the maximum number of elements that you can take from b.
			int32_t high = min((int32_t)k, (int32_t)n_b);
			// checks if the interval correspond with it self (2,2)...
			while (low < high)
			{
				// guesses how many elements are from b, k-mid from c. So if k = 4 and mid = 3 => 3 elements from B, 1 from C. Then check if this is true.
				int32_t mid = (low + high) >> 1;
				// if this is true, then we should consider to take at least one more element from B, one less from C.
				// if not true, then we should slash high, this means that we should take fewer items from B (it was too high) and take more from C then.
				bool cond = b.vd0[mid] <= c.vd0[k - 1 - mid];
				low = cond ? mid + 1 : low;
				high = cond ? high : mid;
			}
			// at this point low is the exact same as high. This is the index of the correct element of B that might be at the index k of the final array
			int32_t i = low;
			// this is the index of C that could be in the index k of the final array
			int32_t j = k - i;
			// this simply check if b[i] < c[j], then writes on a the correct one. if i is not in rage of B then takes C. If j is not in range of C or B[i] is the best one, then take B[i].
			a.vd0[k] = (i < n_b && (j >= n_c || b.vd0[i] <= c.vd0[j])) ? b.vd0[i] : c.vd0[j];
			a.vd1[k] = (i < n_b && (j >= n_c || b.vd0[i] <= c.vd0[j])) ? b.vd1[i] : c.vd1[j];
		}
		base_k += blockDim.x;
	}
	__syncthreads();

	return gwf_intv_merge_adj(total, a);
}

/*
 * Diagonal
 */

// this function checks for if duplicates are going to be inserted in B. If so, then we can process them differently
__device__ __inline__ static int32_t gwf_diag_update(gwf_diag_t *p, uint32_t v, int32_t d, int32_t k, uint32_t x, uint32_t ooo)
{
	const uint64_t vd = gwf_gen_vd(v, d);
	// this checks if a duplicate is going to be inserted
	if (p->vd == vd)
	{
		// in here is chose wich one of the two options is better, the one that is already in B or the one that is give. (better means higher k)
		// the flag here should always be 1 (ooo) when this function is called from gwf_extend()
		p->xo = p->k > k ? p->xo : x << 1 | ooo;

		p->k = p->k > k ? p->k : k;
		return 0;
	}

	return 1;
}

__device__ __inline__ static int gwf_diag_is_sorted(int32_t n_a, const gwf_diag_soa_t a, uint64_t *a_shared)
{

	uint32_t idx = 0;
	uint32_t curr_buf = 0;
	uint32_t next_buf = 32;

	if (idx + threadIdx.x < n_a)
	{
		a_shared[curr_buf + threadIdx.x] = a.vd[idx + threadIdx.x];
	}
	idx = 1;
	uint64_t first_vd = a_shared[0];
	uint64_t mask = 0;
	idx += 32;
	while (idx - 32 < n_a)
	{
		bool in_range_next = idx + threadIdx.x < n_a;
		if (in_range_next)
		{
			a_shared[next_buf + threadIdx.x] = a.vd[idx + threadIdx.x];
		}
		bool in_range_curr = (idx - 32U) + threadIdx.x < n_a;
		const uint64_t curr = in_range_curr ? a_shared[curr_buf + threadIdx.x] : 0ULL;
		// takes the value of thread k - 1.
		uint64_t prev = __shfl_up_sync(0xffffffff, curr, 1);

		prev = (threadIdx.x == 0) ? first_vd : prev;

		mask = __ballot_sync(0xffffffff, in_range_curr && prev > curr);
		if (mask != 0)
		{
			return 0;
		}

		// everyone reads the value of thread 31 (this is the one needed for thread 0 on the next iteration)
		first_vd = __shfl_sync(0xffffffff, curr, 31);

		idx += blockDim.x;

		const int32_t tmp = curr_buf;
		curr_buf = next_buf;
		next_buf = tmp;
	}
	return 1;
}

__device__ __inline__ void sort_gwf_diags_odd_even(gwf_diag_t *__restrict__ c, const int32_t n)
{
	const int32_t tid = threadIdx.x;

	for (int32_t i = 0; i < n; i++)
	{
		int32_t offset = i & 1;

		for (int32_t j = offset + (tid * 2); j < n - 1; j += (THREADS_PER_BLOCK * 2))
		{
			if (c[j].vd > c[j + 1].vd)
			{
				gwf_diag_t temp = c[j];
				c[j] = c[j + 1];
				c[j + 1] = temp;
			}
		}

		__syncthreads();
	}
}

__device__ __inline__ static void sort_gwf_diags_merge(gwf_diag_t *__restrict__ c, int32_t n, gwf_diag_t *__restrict__ tmp){
	gwf_diag_t *src = c;
	gwf_diag_t *dst = tmp;

	for (int32_t width = 1; width < n; width <<= 1)
	{
		for (int32_t left = 0; left < n; left += (width << 1))
		{
			int32_t mid = left + width;
			int32_t right = left + (width << 1);

			if (mid > n)
				mid = n;
			if (right > n)
				right = n;

			const int32_t n1 = mid - left;
			const int32_t n2 = right - mid;
			const int32_t total = n1 + n2;

			for (int32_t out = threadIdx.x; out < total; out += blockDim.x)
			{
				int32_t low = max((int32_t)0, out - n2);
				int32_t high = min(out, n1);

				while (low < high)
				{
					int32_t m = (low + high) >> 1;
					int32_t j = out - 1 - m;

					bool cond = src[left + m].vd <= src[mid + j].vd;

					low = cond ? m + 1 : low;
					high = cond ? high : m;
				}

				int32_t i = low;
				int32_t j = out - i;
				bool take_left = i < n1 && (j >= n2 || src[left + i].vd <= src[mid + j].vd);

				dst[left + out] = take_left ? src[left + i] : src[mid + j];
			}
		}

		__syncthreads();

		gwf_diag_t *t = src;
		src = dst;
		dst = t;

		// __syncthreads();
	}

	if (src != c)
	{
		for (int32_t i = threadIdx.x; i < n; i += blockDim.x)
			c[i] = src[i];

		__syncthreads();
	}
}

__device__ __inline__ void sort_gwf_diags(
	gwf_diag_t *__restrict__ c,
	const int32_t n,
	gwf_diag_t *__restrict__ tmp,
	const int32_t max_tmp)
{
	if (n <= 1)
		return;

	if (n > 64 && n <= max_tmp)
	{
		sort_gwf_diags_merge(c, n, tmp);
		return;
	}

	sort_gwf_diags_odd_even(c, n);
}

__device__ __inline__ void sort_gwf_diags(gwf_diag_t *__restrict__ c, const int32_t n)
{
	sort_gwf_diags_odd_even(c, n);
}

// sort a[]. This uses the gwf_diag_t::ooo field to speed up sorting.
__device__ __inline__ static void gwf_diag_sort(int32_t n_a, gwf_diag_soa_t a, gwf_diag_t *ooo, gwf_diag_t *scratch, int32_t max_scratch){
	int32_t k = 0;
	int32_t n_b;
	gwf_diag_t *b, *c;

	// the one with the o bit high (1) are NOT ordered correclty (are the one that are created).
	// in here there's just a count of how many there are (n_c)

	int32_t n_c = 0;
	while (k < n_a)
	{
		bool in_range = threadIdx.x + k < n_a;
		uint32_t x = in_range ? a.xo[threadIdx.x + k] : 0;
		uint32_t mask = __ballot_sync(0xffffffff, in_range && (x & 1));
		n_c += __popc(mask);
		k += blockDim.x;
	}

	n_b = n_a - n_c;
	b = ooo, c = b + n_b;

	// if the bit is 1 then they are placed in c, otherwise, in b.
	int32_t total_b = 0;
	int32_t total_c = 0;
	k = 0;

	while (k < n_a)
	{

		int32_t current_idx = k + threadIdx.x;
		bool in_range = current_idx < n_a;

		gwf_diag_t x = {};
		if (in_range)
		{
			x.vd = a.vd[current_idx];
			x.k = a.k[current_idx];
			x.xo = a.xo[current_idx];
		}

		bool is_c = in_range && (x.xo & 1);
		bool is_b = in_range && !is_c;

		unsigned mask_c = __ballot_sync(0xffffffff, is_c);
		unsigned mask_b = __ballot_sync(0xffffffff, is_b);

		unsigned int thread_idx_mask = (1U << threadIdx.x) - 1;
		int pos_c = __popc(mask_c & thread_idx_mask);
		int pos_b = __popc(mask_b & thread_idx_mask);

		if (is_c)
		{
			c[total_c + pos_c] = x;
		}
		else if (is_b)
		{
			b[total_b + pos_b] = x;
		}

		total_c += __popc(mask_c);
		total_b += __popc(mask_b);

		k += blockDim.x;
	}

	// c gets sorted

	sort_gwf_diags(c, n_c, scratch, max_scratch);

	// sets the last bit to 0 (0xffffffeU means 1111 1111 1111 1111 .... 1110)
	int32_t base_idx = 0;
	while (base_idx < n_c)
	{
		bool in_range = base_idx + threadIdx.x < n_c;

		if (in_range)
		{
			c[base_idx + threadIdx.x].xo &= 0xfffffffeU;
		}
		base_idx += blockDim.x;
	}
	__syncthreads();

	// just like earlier in the process we sort b and c
	int32_t total = n_b + n_c;
	int32_t base_k = 0;

	// merge path algorithm. k is the index in wich a certain element should appear in the final array
	while (base_k < total)
	{
		int32_t k = base_k + threadIdx.x;
		bool in_range = k < total;
		if (in_range)
		{
			// this is the minimum elements that you'll need to take from b. If k = 100 and n_c is small (20) then you'll need 80 elements from b
			int32_t low = max((int32_t)0, (int32_t)(k - n_c));
			// this is the maximum number of elements that you can take from b.
			int32_t high = min((int32_t)k, (int32_t)n_b);
			// checks if the interval correspond with it self (2,2)...
			while (low < high)
			{
				// guesses how many elements are from b, k-mid from c. So if k = 4 and mid = 3 => 3 elements from B, 1 from C. Then check if this is true.
				int32_t mid = (low + high) >> 1;
				// if this is true, then we should consider to take at least one more element from B, one less from C.
				// if not true, then we should slash high, this means that we should take fewer items from B (it was too high) and take more from C then.
				bool cond = b[mid].vd <= c[k - 1 - mid].vd;
				low = cond ? mid + 1 : low;
				high = cond ? high : mid;
			}
			// at this point low is the exact same as high. This is the index of the correct element of B that might be at the index k of the final array
			int32_t i = low;
			// this is the index of C that could be in the index k of the final array
			int32_t j = k - i;
			// this simply check if b[i] < c[j], then writes on a the correct one. if i is not in rage of B then takes C. If j is not in range of C or B[i] is the best one, then take B[i].
			gwf_diag_t best = (i < n_b && (j >= n_c || b[i].vd <= c[j].vd)) ? b[i] : c[j];

			a.vd[k] = best.vd;
			a.k[k] = best.k;
			a.xo[k] = best.xo;
		}
		base_k += blockDim.x;
	}
	// some k needs more loops then others.
	__syncthreads();
}

__device__ static int32_t gwf_diag_dedup(int32_t n_a, gwf_diag_soa_t a, gwf_diag_t *ooo, gwf_diag_t *scratch, int32_t max_scratch, uint64_t *a_shared){
	int32_t i, n, st;
	// if a is not sorted, sort it
	if (!gwf_diag_is_sorted(n_a, a, a_shared))
		gwf_diag_sort(n_a, a, ooo, scratch, max_scratch);

	for (i = 1, st = 0, n = 0; i <= n_a; ++i)
	{
		// this if is triggered when there's a jump, so if a different vd is found
		if (i == n_a || a.vd[i] != a.vd[st])
		{
			// st is for the start of a block that has the same vd.
			int32_t j, max_j = st;
			// here we just find out the best k among [st, i-1] (those that have the same vd)
			if (st + 1 < i)
			{
				for (j = st + 1; j < i; ++j)
				{ // choose the far end (i.e. the wavefront)
					if (a.k[max_j] < a.k[j])
						max_j = j;
				}
			}

			// this is a similar way of handling things of the one used for intv earlier. This just scraps away everything and keeps in position n the best diagonal.
			a.vd[n] = a.vd[max_j];
			a.k[n] = a.k[max_j];
			a.xo[n] = a.xo[max_j];
			n++;
			// reset starting point
			st = i;
		}
	}
	// this returns the ammount of best states that currently survived in a.
	return n;
}

// remove diagonals not on the wavefront
// use forbidden bands to remove diagonals not on the wavefront
__device__ __inline__ static int32_t gwf_mixed_dedup(int32_t n_a, gwf_diag_soa_t a, int32_t n_b, gwf_intv_soa_t b)
{
	int32_t k_write = 0;
	int32_t k = 0;

	while (k < n_a)
	{
		int32_t idx = k + threadIdx.x;
		bool in_range = idx < n_a;

		gwf_diag_t curr_a = {};
		if (in_range)
		{
			curr_a.vd = a.vd[idx];
			curr_a.k = a.k[idx];
			curr_a.xo = a.xo[idx];
		}

		bool keep = in_range;
		if (in_range)
		{
			int low = 0, high = n_b - 1;
			while (low <= high)
			{
				int32_t mid = (low + high) / 2;
				if (curr_a.vd >= b.vd0[mid] && curr_a.vd < b.vd1[mid])
				{
					keep = false;
					break;
				}
				if (curr_a.vd < b.vd0[mid])
					high = mid - 1;
				else
					low = mid + 1;
			}
		}

		uint32_t keep_mask = __ballot_sync(0xffffffff, keep);
		int32_t total_to_write = __popc(keep_mask);

		if (keep)
		{
			uint32_t lane_mask = (1U << threadIdx.x) - 1;
			int32_t pos = __popc(keep_mask & lane_mask);
			int32_t write_idx = k_write + pos;

			a.vd[write_idx] = curr_a.vd;
			a.k[write_idx] = curr_a.k;
			a.xo[write_idx] = curr_a.xo;
		}

		k_write += total_to_write;
		k += blockDim.x;
	}

	return k_write;
}
/*
 * Core GWFA routine
 */
__device__ __inline__ static void static_intv_copy(gwf_intv_soa_t dst_ptr, counters &cnt, int32_t max_dst, const gwf_intv_soa_t src_ptr, int32_t n_src)
{

	if (n_src > max_dst)
	{
		cnt.n_swap = 0;
		return;
	}

	int32_t base_k = 0;

	while (base_k < n_src)
	{
		int32_t k = base_k + threadIdx.x;
		bool in_range = (k < n_src);
		if (in_range)
		{
			dst_ptr.vd0[k] = src_ptr.vd0[k];
			dst_ptr.vd1[k] = src_ptr.vd1[k];
		}

		base_k += blockDim.x;
	}

	__syncthreads();

	cnt.n_swap = n_src;
}

__device__ __inline__ static int32_t gwf_dedup(gwf_edbuf_t *buf, int32_t n_a, gwf_diag_soa_t a, counters &cnt, uint64_t *diag_shared)
{

	if (cnt.n_intv + cnt.n_tmp > 0)
	{
		// tmp contains the new states, it just gets ordered with radix sort
		if (!gwf_intv_is_sorted(cnt.n_tmp, buf->tmp, diag_shared))
			fast_parallel_sort(buf->tmp.vd0, buf->tmp.vd1, cnt.n_tmp, buf->swap.vd0, buf->swap.vd1, (int32_t)buf->max_swap);

		// copies the content of intv into swap.
		// intv contains the memory of the intervals of the other states before the current one. During s = 0, this call is useless and intv will be populated with just tmp of round 0
		static_intv_copy(buf->swap, cnt, buf->max_swap, buf->intv, cnt.n_intv);

		// intv is now complete (should be very short) and is correctly ordered. The elements contains the intervals of close diags.
		cnt.n_intv = gwf_intv_merge2(buf->intv, cnt.n_swap, buf->swap, cnt.n_tmp, buf->tmp);
	}

	// doese a lot of things, lastly it removes duplicates that are behind and returns the new a lenght.
	n_a = gwf_diag_dedup(n_a, a, buf->ooo, buf->B, (int32_t)buf->max_B, diag_shared);

	if (cnt.n_intv > 0)
		// filters a based on intv.
		n_a = gwf_mixed_dedup(n_a, a, cnt.n_intv, buf->intv);
	return n_a;
}

__device__ __inline__ static int32_t gwf_extend1(int32_t d, int32_t k, int32_t vl, const char *__restrict__ ts, int32_t ql, const char *__restrict__ qs)
{

	const uint32_t max_k = (ql - d < vl ? ql - d : vl) - 1;
	const char *const ts_ = ts + 1;
	const char *const qs_ = qs + d + 1;

	// __shared__ uint8_t ts_shared[64];
	// __shared__ uint8_t qs_shared[64];

	while (k + 31 < max_k)
	{

		uint32_t lane = threadIdx.x;
		uint8_t x = *(ts_ + k + lane);
		uint8_t y = *(qs_ + k + lane);

		uint32_t mask;
		mask = __ballot_sync(0xffffffff, x != y);

		if (mask != 0)
		{
			return (k + (__ffs(mask) - 1));
		}
		k += blockDim.x;
	}

	while (k < max_k && *(ts_ + k) == *(qs_ + k))
	{
		++k;
	}

	return k;
}

__device__ __inline__ static gwf_diag_t *static_A_pushp(gwf_edbuf_t *buf, counters &cnt)
{

	gwf_diag_t *__restrict__ A = buf->A;
	const size_t max_A = buf->max_A;
	size_t tail_A = buf->tail_A;

	gwf_diag_t *__restrict__ p = &A[tail_A];
	tail_A++;
	if (tail_A >= max_A)
		tail_A = 0;
	cnt.n_A++;

	buf->tail_A = tail_A;

	return p;
}

__device__ __inline__ static gwf_diag_t static_A_shift(gwf_edbuf_t *buf, counters &cnt)
{
	gwf_diag_t *__restrict__ A = buf->A;
	const size_t max_A = buf->max_A;
	size_t head_A = buf->head_A;

	if (cnt.n_A == 0)
		return (gwf_diag_t){0};

	gwf_diag_t t = A[head_A];

	head_A++;
	if (head_A >= max_A)
		head_A = 0;
	cnt.n_A--;

	buf->head_A = head_A;
	return t;
}

__device__ __inline__ static void static_B_pushp(gwf_edbuf_t *buf, int32_t v, int32_t d, int32_t k, uint32_t xo, int32_t ooo, counters &cnt)
{

	gwf_diag_t *__restrict__ p = &buf->B[cnt.n_B];
	p->vd = gwf_gen_vd(v, d);
	p->k = k;
	p->xo = xo << 1 | ooo;

	cnt.n_B++;
}


__device__ __inline__ static void gwf_ed_extend_batch(const gwf_graph_t *g, int32_t ql, const char *q, int32_t n, gwf_diag_soa_t a, int32_t offset, gwf_edbuf_t *buf, counters &cnt)
{
	int32_t j, m;
	uint32_t v = (uint32_t)(a.vd[offset] >> 32);
	uint32_t vl = g->len[v];
	const char *ts = g->seq[v];
	gwf_diag_t *b;

	gwf_diag_t *B = buf->B;

	// this extends all of the diagonals n in the same vertex v. The number n is (i - x) from gwf_ed_extend.
	for (j = 0; j < n; ++j)
	{
		int32_t k;
		k = gwf_extend1((int32_t)a.vd[j + offset] - GWF_DIAG_SHIFT, a.k[j + offset], vl, ts, ql, q);
		// this is the number of matches found during gwf_extend1. The << 2 is for leaving the first bit (flag) unchanged.
		//  It is an update of the antidiagonal.
		a.xo[j + offset] += (k - a.k[j + offset]) << 2;
		a.k[j + offset] = k;
	}

	// b is taking care of the new diagonals that will be analyzed in the next iteration s + 1
	b = &B[cnt.n_B];

	// b[0] is the deletion case (character in the graph but not in the query)
	// downshifting the diagonal for the deletion
	b[0].vd = a.vd[0 + offset] - 1;
	// increase the antidiagonal by 1 (2 == 1<<1). Ofc this is because we need to advance by one, zero on the query. Same concept applies for k
	b[0].xo = a.xo[0 + offset] + 2;
	b[0].k = a.k[0 + offset] + 1;

	// b[1] is the mismatch so it's in the same diagonal as the one that originates the mismatch. It can be reached by both an a[0] mismatch or an a[1] deletion.
	//  we are looking at a border decided by the contiguos diagonals that are given to this funcion. For this specific case there's no way for "a[-1]" = b[1] to exist (insertion)
	b[1].vd = a.vd[0 + offset];
	// for the antidiagonal we chose the one that is the furthest (1+1 for the mismatch (1 for the query and 1 for the graph) and (1 + 0) for a deletion (1 for the query, 0 for the graph))
	b[1].xo = n == 1 || a.k[0 + offset] > a.k[1 + offset] ? a.xo[0 + offset] + 4 : a.xo[1 + offset] + 2;
	// here k advances by 1 even in case of a deletion. This is because the diagonal is a - 1 and k is + 1. So in the end the query stands still (i = k - d). In case of a mismatch is correct.
	b[1].k = (n == 1 || a.k[0 + offset] > a.k[1 + offset] ? a.k[0 + offset] : a.k[1 + offset]) + 1;

	// the following calculations are done by considering the diagonal of a[j-1] meaning the d-1 diagonal.
	// b is mapped to a +1 because the previous statement of b[0] and b[1]. Of course this means that there will be a +1 state.

	// sidenote on n values: they range from 1 to 64. Mainly [1,5] tho so not much to do here
	// here is decided which situation is to pick. A mismatch, deletion or insertion from a[j-1 / j / j+1] to assign to b[j+1].
	for (j = 1; j < n - 1; ++j)
	{
		// assuming that the best point way to reach b[j+1] is an insertion: k = k, antidiagonal + 1
		uint32_t x = a.xo[j - 1 + offset] + 2;
		int32_t k = a.k[j - 1 + offset];

		// was a mismatch better? +2 for the antidiagonal and +1 k
		x = k > a.k[j + offset] + 1 ? x : a.xo[j + offset] + 4;
		k = k > a.k[j + offset] + 1 ? k : a.k[j + offset] + 1;

		// is a deletion better? +1 for the antidiagonal and +1 k
		x = k > a.k[j + 1 + offset] + 1 ? x : a.xo[j + 1 + offset] + 2;
		k = k > a.k[j + 1 + offset] + 1 ? k : a.k[j + 1 + offset] + 1;

		// set b[j+1] with the best result
		b[j + 1].vd = a.vd[j + offset], b[j + 1].k = k, b[j + 1].xo = x;
	}
	// this is for the right border. Just like the left border with 0 and 1, here we calculate if it's best an insertion or a mismatch.
	if (n >= 2)
	{
		b[n].vd = a.vd[n - 1 + offset];
		b[n].xo = a.k[n - 2 + offset] > a.k[n - 1 + offset] + 1 ? a.xo[n - 2 + offset] + 2 : a.xo[n - 1 + offset] + 4;

		b[n].k = a.k[n - 2 + offset] > a.k[n - 1 + offset] + 1 ? a.k[n - 2 + offset] : a.k[n - 1 + offset] + 1;
	}
	b[n + 1].vd = a.vd[n - 1 + offset] + 1;
	b[n + 1].xo = a.xo[n - 1 + offset] + 2;

	b[n + 1].k = a.k[n - 1 + offset];

	for (j = 0; j < n; ++j)
	{
		int32_t idx = j + offset;

		int32_t curr_k = a.k[idx];
		uint64_t curr_vd = a.vd[idx];

		if (curr_k == vl - 1 || (int32_t)curr_vd - GWF_DIAG_SHIFT + curr_k == ql - 1)
		{
			a.xo[idx] |= 1;

			gwf_diag_t *A_pushed = static_A_pushp(buf, cnt);
			if (A_pushed)
			{
				A_pushed->vd = curr_vd;
				A_pushed->k = curr_k;
				A_pushed->xo = a.xo[idx];
			}
		}
	}

	// this loop goes through every new generated state (n+2) and decides what to do with every state. This rewrites b.
	for (j = 0, m = 0; j < n + 2; ++j)
	{
		gwf_diag_t *p = &b[j];
		int32_t d = (int32_t)p->vd - GWF_DIAG_SHIFT;

		// if the generate state is still in the same vertex and the query is not done, then it is to keep
		if (d + p->k < ql && p->k < vl)
		{
			b[m++] = *p;

			// if the vertex is done but not the query, then we push this state in the tmp buffer.
		}
		else if (p->k == vl)
		{
			// gwf_intv_t *q;
			// q = static_tmp_pushp(buf, cnt);
			// if(q){
			//     //vd0 is filled normally, vd1 is given to create an interval. This doesn't include vd1 as a state that reached the end of the vertex because the interval is [vd0, vd1).
			//     //this is just because further function will need this.
			// 	q->vd0 = gwf_gen_vd(v, d), q->vd1 = q->vd0 + 1;
			// }
			if (cnt.n_tmp < buf->max_tmp)
			{
				const uint64_t val = gwf_gen_vd(v, d);
				buf->tmp.vd0[cnt.n_tmp] = val;
				buf->tmp.vd1[cnt.n_tmp] = val + 1;
				cnt.n_tmp++;
			}
		}
	}
	cnt.n_B += m;
}

__device__ __inline__ static void static_set64_put(uint64_t *ha, size_t ha_mask, uint64_t key, int *absent)
{
	if (key == 0)
	{
		*absent = 0;
		return;
	}

	const size_t cap = ha_mask + 1;
	size_t i = (key * 0x45d9f3b) & ha_mask;

	for (size_t probes = 0; probes < cap; ++probes)
	{
		uint64_t cur = ha[i];

		if (cur == key)
		{
			*absent = 0;
			return;
		}

		if (cur == 0)
		{
			ha[i] = key;
			*absent = 1;
			return;
		}

		i = (i + 1) & ha_mask;

		if ((probes & 0xfffff) == 0xfffff)
			gwf_debug_hash_probe_mark((int32_t)probes);
	}

	/*
		Hash table full / degenerate.
		For now, do not spin forever.
		This may affect correctness, but it lets the program terminate
		and confirms the diagnosis.
	*/
	*absent = 0;
	gwf_debug_hash_probe_mark((int32_t)cap);
}

__device__ __inline__ static void gwf_ed_extend(gwf_edbuf_t *buf, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v1, uint32_t max_lag, int32_t *end_v, int32_t *end_off, counters &cnt, uint64_t *diag_shared, int32_t score_s)
{

	int32_t i, x, do_dedup = 1;

	cnt.n_A = 0;
	buf->head_A = 0;
	buf->tail_A = 0;

	cnt.n_B = 0;

	gwf_diag_soa_t a = buf->a_buf;
	int32_t n = cnt.n_a;

	// end_v is going to memorize the correct vertex once the algorithm is done. End_off is the offset (k)
	*end_v = *end_off = -1;

	gwf_debug_mark(GWF_DBG_STAGE_EXTEND_ENTER, score_s, cnt.n_a, cnt.n_A, cnt.n_B, cnt.n_tmp, *end_v, *end_off, ql);

	cnt.n_tmp = 0;

	// batch extension for diagonals that are in the same vertex. This loop groups the diags that are close and in the same vertex
	for (x = 0, i = 1; i <= (int32_t)n; ++i)
	{
		// this checks if the diag in two contiguos states are close to each other, if not, extend. (meaning that there's a hole / different vertex). a is ordered by vertex and diagonals.
		if (i == (int32_t)n || a.vd[i] != a.vd[i - 1] + 1)
		{
			// i - x is the number of close togheter diagonal. Of course those are in the same vertex
			gwf_ed_extend_batch(g, ql, q, i - x, buf->a_buf, x, buf, cnt);
			// x is updated for finding the next close thogeter diagonals
			x = i;
		}
	}
	// if A is empty this means that there are no active states that are at the end of a vertex / that have completed the query. No need to clean the redundant states
	if (cnt.n_A == 0)
		do_dedup = 0;

	// this loop adds states to A because it decides how to propagate them through other vertices. This is why the flag ooo could be 0 here. For the same reason, gwf_extend1 is needed
	// because it'll extend the NEW states in A. Ofc the first part of this loop is useless for the states that are generated in gwf_ed_extend_batch()
#ifdef GWF_ENABLE_DEBUG_LOG
	int32_t debug_a_pops = 0;
#endif
	while (cnt.n_A > 0)
	{
#ifdef GWF_ENABLE_DEBUG_LOG
		++debug_a_pops;
		if ((debug_a_pops & 0x3fff) == 0)
			gwf_debug_mark(GWF_DBG_STAGE_EXTEND_A_LOOP, score_s, cnt.n_a, cnt.n_A, cnt.n_B, cnt.n_tmp, *end_v, *end_off, ql);
#endif
		gwf_diag_t t;
		uint32_t x0;
		int32_t ooo, v, d, k, i, vl;

		// takes one of the current states in A.
		t = static_A_shift(buf, cnt);
		// ooo is the flag of xo, usefull for the new states that will be added later to A
		ooo = t.xo & 1;
		// vertex
		v = t.vd >> 32;
		// diagonal
		d = (int32_t)t.vd - GWF_DIAG_SHIFT;
		// wavefront position on the vertex
		k = t.k;
		// vertex length
		vl = g->len[v];

		// extends the current diagonal. This is useless for the first iteration of elements that already were in A.
		// this call is used to extend those states that already passed a vertex and have a perfect match with the following character of the query
		k = gwf_extend1(d, k, vl, g->seq[v], ql, q);

		// query position
		i = k + d;

		// current antidiagonal reached with the extend. The ooo bit is not considered here. x0 contains just the antidiagonal
		x0 = (t.xo >> 1) + ((k - t.k) << 1);

		// if we are in the middle of the vertex (this situation will present itself only before the next condition turns out to be true)
		if (k + 1 < vl && i + 1 < ql)
		{

			// push 1 and push 2 decide what operation should be done. With gwf_diag_update, we update the current element in B to be the best possibile by confronting it to the
			// given one (present in A). This is because, if the diags are on the same vertex, then the product of the missmatch and the delition will already be in B!.
			// If the current extention of the state is already present in B, then push1/2 will be 0, so no further pushing, just updating.
			int32_t push1 = 1, push2 = 1;
			if (cnt.n_B >= 2)
				push1 = gwf_diag_update(&buf->B[cnt.n_B - 2], v, d - 1, k + 1, x0 + 1, ooo);
			if (cnt.n_B >= 1)
				push2 = gwf_diag_update(&buf->B[cnt.n_B - 1], v, d, k + 1, x0 + 2, ooo);
			if (push1)
				static_B_pushp(buf, v, d - 1, k + 1, x0 + 1, 1, cnt);
			if (push2 || push1)
				static_B_pushp(buf, v, d, k + 1, x0 + 2, 1, cnt);
			// the last push is always done because it can't be on the same diagonal as the one of an element in B (insertion)
			static_B_pushp(buf, v, d + 1, k, x0 + 1, ooo, cnt);

			// reaching the end of the vertex but not the end of query. This is the first brach that the originale states in A follow.
		}
		else if (i + 1 < ql)
		{
			// vertex ofset (references g->arc) of the close vertexes. High 32 bits
			int32_t ov = g->aux[v] >> 32;
			// numer of close vertex that are close to v. Low 32 bits
			int32_t nv = (int32_t)g->aux[v];
			int32_t n_ext = 0;
			// gwf_intv_t *p;

			// here is added a new state to tmp. For the first iteration (old A content) this is redundant (already done in extend_batch). This is still necessary because a current state
			// could reach the end of a new vertex.
			//  p = static_tmp_pushp(buf, cnt);
			//  if(p){
			//  	//vd1 is for cleanup purposes later on. as stated in gwf_extend_batch(), for the states in tmp, an interval is needed
			//  	p->vd0 = gwf_gen_vd(v, d), p->vd1 = p->vd0 + 1;
			//  }
			if (cnt.n_tmp < buf->max_tmp)
			{
				uint64_t val = gwf_gen_vd(v, d);
				buf->tmp.vd0[cnt.n_tmp] = val;
				buf->tmp.vd1[cnt.n_tmp] = val + 1;
				cnt.n_tmp++;
			}

			// traversing v's neighbors
			for (int32_t j = 0; j < nv; ++j)
			{
				// w is a v neighbour (ov was the location offset)
				uint32_t w = (uint32_t)g->arc[ov + j].a;
				// ol is the overlap. Sometimes the algorithm doesn't start from the first character of the vertex because they might be already covered by another vertex (graph structure)
				int32_t ol = g->arc[ov + j].o;

				// test if it's the first time arriving in this vertex from exactly this point (ID == (vertex w)|(query position i + 1))
				// made because of cyclic graphs
				int absent;
				static_set64_put(buf->ha, buf->ha_mask, (uint64_t)w << 32 | (i + 1), &absent);

				// if there is a free extention possibile (first caracter is the same as the "first" (ol ofset) character of the vertex)
				if (q[i + 1] == g->seq[w][ol])
				{
					// number of neighbours extensions
					++n_ext;
					if (absent)
					{
						gwf_diag_t *p;
						p = static_A_pushp(buf, cnt);
						if (p)
						{
							// as d = i - k, we have that i = i+1 and k = ol. x0 + 2 is becuase there is a direct match so 1 for the query and 1 for k (ol)
							p->vd = gwf_gen_vd(w, i + 1 - ol), p->k = ol, p->xo = (x0 + 2) << 1 | 1;
						}
					}
					// if there's no match but the propagation is valid then we have to generate the two new state. Only two because if there's an insertion, then the graph should remain still
					// meaning that the vertex should not be passed
				}
				else if (absent)
				{
					// deletion
					static_B_pushp(buf, w, i - ol, ol, x0 + 1, 1, cnt);
					// mismatch
					static_B_pushp(buf, w, i + 1 - ol, ol, x0 + 2, 1, cnt);
				}
			} // if there is an insertion. So, if there are no neighbours OR if the number of extensions done is lower than the numer of neighbours.
			// the second condition is made for deleting redundances because, if each state had a perfect match, then there's no point in considering an insertion that will be overrided later on for sure
			if (nv == 0 || n_ext != nv)
				static_B_pushp(buf, v, d + 1, k, x0 + 1, 1, cnt);

			// v1 = -1 always ==> we don't have a specific vertex to reach.
			// this is the last case: if the query is done (i + 1 == ql)
		}
		else if (v1 < 0 || (v == v1 && k + 1 == vl))
		{
			*end_v = v, *end_off = k;
			cnt.n_A = 0;
			return;
		}
		else if (k + 1 < vl)
		{ // i + 1 == ql; reaching the end of the query but not the end of the vertex
			// here we add a delition. Most likely it'll result in a duplicate.
			static_B_pushp(buf, v, d - 1, k + 1, x0 + 1, ooo, cnt);

			// if we reach the end of the query and vertex but not reached v1. Not happening ==> v1 is -1.
		}
		else if (v != v1)
		{
			int32_t ov = g->aux[v] >> 32, nv = (int32_t)g->aux[v], j;

			for (j = 0; j < nv; ++j)
			{
				uint32_t w = (uint32_t)g->arc[ov + j].a;
				int32_t ol = g->arc[ov + j].o;
				static_B_pushp(buf, w, i - ol, ol, x0 + 1, 1, cnt); // deleting the first base on the next vertex
			}
		} // else assert(0); // should never come here
	}

	gwf_debug_mark(GWF_DBG_STAGE_COPY_B_TO_A, score_s, cnt.n_a, cnt.n_A, cnt.n_B, cnt.n_tmp, *end_v, *end_off, ql);

	// Exchanging the buffers: a <- B, B is empty.
	// The first barrier is required because B is produced cooperatively by the block.
	__syncthreads();

	for (int32_t j = threadIdx.x; j < cnt.n_B; j += blockDim.x)
	{
		gwf_diag_t temp = buf->B[j];

		buf->a_buf.vd[j] = temp.vd;
		buf->a_buf.k[j] = temp.k;
		buf->a_buf.xo[j] = temp.xo;
	}

	// Ensure all lanes completed the copy before a_buf is consumed.
	__syncthreads();

	cnt.n_a = cnt.n_B;
	cnt.n_B = 0;

	int32_t current_n = cnt.n_a;

	// removes from "a" duplicates and slow diagonals that happen to reach the same diagonal while exiting a vertex as an older one (so the older has a much more convinient k)
	if (do_dedup)
		current_n = gwf_dedup(buf, current_n, buf->a_buf, cnt, diag_shared);

	gwf_debug_mark(GWF_DBG_STAGE_DEDUP_DONE, score_s, current_n, cnt.n_A, cnt.n_B, cnt.n_tmp, *end_v, *end_off, ql);

	// not happening, max_lag is set to 0 (no boundaries)
	//  if (max_lag > 0) current_n = gwf_prune(current_n, buf->a_buf, max_lag);

	cnt.n_a = current_n;
}

__device__ int32_t gwf_ed(const gwf_graph_t *g, int32_t ql, const char *q, int32_t v0, int32_t v1, uint32_t max_lag, gwf_edbuf_t *buf, uint64_t *diag_shared)
{

	int32_t s = 0;
	int32_t end_v = -1;
	int32_t end_off = -1;

	gwf_diag_soa_t const a = buf->a_buf;

	counters cnt = {};

	a.vd[0] = gwf_gen_vd(v0, 0);

	a.k[0] = -1;
	a.xo[0] = 0;

	cnt.n_a = 1;

	gwf_debug_mark(GWF_DBG_STAGE_GWF_START, s, cnt.n_a, cnt.n_A, cnt.n_B, cnt.n_tmp, end_v, end_off, ql);

	// expand until there's at least one state in the queue a
	while (cnt.n_a > 0)
	{

		// finds out the new reachable states at s+1 distance from the current set.
#ifdef GWF_ENABLE_DEBUG_LOG
		unsigned long long dbg_stride = gwf_debug_stride == 0ULL ? 1ULL : gwf_debug_stride;
		if (((unsigned long long)(long long)s % dbg_stride) == 0ULL)
			gwf_debug_mark(GWF_DBG_STAGE_BEFORE_EXTEND, s, cnt.n_a, cnt.n_A, cnt.n_B, cnt.n_tmp, end_v, end_off, ql);
#endif

		gwf_ed_extend(buf, g, ql, q, v1, max_lag, &end_v, &end_off, cnt, diag_shared, s);

		gwf_debug_mark(GWF_DBG_STAGE_ITERATION_DONE, s, cnt.n_a, cnt.n_A, cnt.n_B, cnt.n_tmp, end_v, end_off, ql);

		if (end_off >= 0 || cnt.n_a == 0)
			break;
		++s;

	}


	gwf_debug_mark(GWF_DBG_STAGE_GWF_DONE, s, cnt.n_a, cnt.n_A, cnt.n_B, cnt.n_tmp, end_v, end_off, ql);

	return (end_v >= 0 ? s : -1);
}
