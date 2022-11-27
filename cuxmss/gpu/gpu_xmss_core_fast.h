#include "gpu_common.h"

typedef struct {
	u8 h;
	u64 next_idx;
	u8 stackusage;
	u8 completed;
	u8 *node;
} treehash_inst;

typedef struct {
	u8 *stack;
	u32 stackoffset;
	u8 *stacklevels;
	u8 *auth;
	u8 *keep;
	treehash_inst *treehash;
	u8 *retain;
	u32 next_leaf;
} bds_state;

__device__ void dev_xmssmt_serialize_state(u8 *sk, bds_state *states);

__device__ void dev_xmssmt_deserialize_state(bds_state *states,
					     u8 **wots_sigs,  u8 *sk);

__device__ void dev_xmss_serialize_state(u8 *sk, bds_state *state);

__device__ void dev_xmss_deserialize_state(bds_state *state, u8 *sk);

__device__ void dev_deep_state_swap(bds_state *a, bds_state *b);

__device__ void dev_treehash_init(u8 *node, int height, int index,
				  bds_state *state, const u8 *sk_seed,
				  const u8 *pub_seed, const u32 addr[8]);

__device__ void dev_treehash_init_parallel_5(u8 *node, int height, int index,
					     bds_state *state, const u8 *sk_seed,
					     const u8 *pub_seed, const u32 addr[8]);

__device__ void dev_treehash_init_parallel_10(u8 *node, int height, int index,
					      bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8]);

__device__ void dev_treehash_init_parallel_10_1(u8 *node, int height, int index,
						bds_state *state, const u8 *sk_seed,
						const u8 *pub_seed, const u32 addr[8]);

__device__ void dev_treehash_init_parallel_10_2(u8 *node, int height, int index,
						bds_state *state, const u8 *sk_seed,
						const u8 *pub_seed, const u32 addr[8]);

__device__ void dev_treehash_init_parallel_16(u8 *node, int height, int index,
					      bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8]);

__device__ void dev_treehash_init_parallel_20(u8 *node, int height, int index,
					      bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8]);

__device__ void dev_treehash_update(treehash_inst *treehash, bds_state *state,
				    const u8 *sk_seed, const u8 *pub_seed, const u32 addr[8]);

__device__ char dev_bds_treehash_update(bds_state *state, u32 updates,
					const u8 *sk_seed, u8 *pub_seed, const u32 addr[8]);

__device__ char dev_bds_treehash_update_parallel(bds_state *state, u32 updates,
						 const u8 *sk_seed, u8 *pub_seed, const u32 addr[8]);

__device__ char dev_bds_state_update(bds_state *state, const u8 *sk_seed,
				     const u8 *pub_seed, const u32 addr[8]);

__device__ char dev_bds_state_update_parallel(bds_state *state, const u8 *sk_seed,
					      const u8 *pub_seed, const u32 addr[8]);

__device__ void dev_bds_round(bds_state *state, const u64 leaf_idx,
			      const u8 *sk_seed, const u8 *pub_seed, u32 addr[8]);

__device__ void dev_bds_round_parallel(bds_state *state, const u64 leaf_idx,
				       const u8 *sk_seed, const u8 *pub_seed, u32 addr[8]);
