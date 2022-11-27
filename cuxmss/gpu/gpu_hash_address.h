#include "gpu_common.h"

#define XMSS_ADDR_TYPE_OTS 0
#define XMSS_ADDR_TYPE_LTREE 1
#define XMSS_ADDR_TYPE_HASHTREE 2

__device__ void dev_set_layer_addr(u32 addr[8], u32 layer);

__device__ void dev_set_tree_addr(u32 addr[8], u64 tree);

__device__ void dev_set_type(u32 addr[8], u32 type);

__device__ void dev_set_key_and_mask(u32 addr[8], u32 key_and_mask);

__device__ void dev_copy_subtree_addr(u32 out[8], const u32 in[8]);

/* These functions are used for OTS addresses. */

__device__ void dev_set_ots_addr(u32 addr[8], u32 ots);

__device__ void dev_set_chain_addr(u32 addr[8], u32 chain);

__device__ void dev_set_hash_addr(u32 addr[8], u32 hash);

/* This function is used for L-tree addresses. */

__device__ void dev_set_ltree_addr(u32 addr[8], u32 ltree);

/* These functions are used for hash tree addresses. */

__device__ void dev_set_tree_height(u32 addr[8], u32 tree_height);

__device__ void dev_set_tree_index(u32 addr[8], u32 tree_index);
