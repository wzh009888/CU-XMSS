#include "all_option.h"

// only include typedef and define
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
#define HOST_2_DEVICE cudaMemcpyHostToDevice
#define DEVICE_2_HOST cudaMemcpyDeviceToHost

// for test
#define XMSS_MLEN 32

/* These are merely internal identifiers for the supported hash functions. */
#define XMSS_SHA2 0
#define XMSS_SHAKE128 1
#define XMSS_SHAKE256 2

/* for constant memory */
extern u32 this_K256[64];
__constant__ extern u32 K256[64];

extern u64 this_K512[80];
__constant__ extern u64 K512[80];

extern u64 this_KeccakF_RoundConstants[24];
__constant__ extern u64 KeccakF_RoundConstants[24];

extern unsigned char this_rhotates[5][5];
__constant__ extern unsigned char rhotates[5][5];

extern unsigned long this_iotas[24];
__constant__ extern unsigned long iotas[24];

/* global varient for testing time */
extern struct timespec g_start, g_stop;
extern double g_result;

#define CHECK(call) \
	if ((call) != cudaSuccess) { \
		cudaError_t err = cudaGetLastError(); \
		cerr << "CUDA error calling \""#call "\", code is " << err << endl; }

u64 cpucycles(void);
int cmp_llu(const void *a, const void *b);
u64 median(u64 *l, u64 llen);
u64 average(u64 *t, u64 tlen);
void print_results(u64 *t, u64 tlen);

/*
 * XMSSMT SHA2 192
 */
#ifdef XMSSMT_SHA2_20_2_192
#define XMSS_VARIANT              "XMSSMT-SHA2_20/2_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         2
#endif /* ifdef XMSSMT_SHA2_20_2_192 */

#ifdef XMSSMT_SHA2_20_4_192
#define XMSS_VARIANT              "XMSSMT-SHA2_20/4_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         4
#endif /* ifdef XMSSMT_SHA2_20_4_192 */

#ifdef XMSSMT_SHA2_40_2_192
#define XMSS_VARIANT              "XMSSMT-SHA2_40/2_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         2
#endif /* ifdef XMSSMT_SHA2_40_2_192 */

#ifdef XMSSMT_SHA2_40_4_192
#define XMSS_VARIANT              "XMSSMT-SHA2_40/4_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         4
#endif /* ifdef XMSSMT_SHA2_40_4_192 */

#ifdef XMSSMT_SHA2_40_8_192
#define XMSS_VARIANT              "XMSSMT-SHA2_40/8_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         8
#endif /* ifdef XMSSMT_SHA2_40_8_192 */

#ifdef XMSSMT_SHA2_60_3_192
#define XMSS_VARIANT              "XMSSMT-SHA2_60/3_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         3
#endif /* ifdef XMSSMT_SHA2_60_6_192 */

#ifdef XMSSMT_SHA2_60_6_192
#define XMSS_VARIANT              "XMSSMT-SHA2_60/6_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         6
#endif /* ifdef XMSSMT_SHA2_60_6_192 */

#ifdef XMSSMT_SHA2_60_12_192
#define XMSS_VARIANT              "XMSSMT-SHA2_60/12_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         12
#endif /* ifdef XMSSMT_SHA2_60_12_192 */

/***********************************
* XMSSMT SHAKE256 192
***********************************/
#ifdef XMSSMT_SHAKE256_20_2_192
#define XMSS_VARIANT              "XMSSMT-SHAKE256_20/2_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         2
#endif /* ifdef XMSSMT_SHAKE256_20_2_192 */

#ifdef XMSSMT_SHAKE256_20_4_192
#define XMSS_VARIANT              "XMSSMT-SHAKE256_20/4_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         4
#endif /* ifdef XMSSMT_SHAKE256_20_4_192 */

#ifdef XMSSMT_SHAKE256_40_2_192
#define XMSS_VARIANT              "XMSSMT-SHAKE256_40/2_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         2
#endif /* ifdef XMSSMT_SHAKE256_40_4_192 */

#ifdef XMSSMT_SHAKE256_40_4_192
#define XMSS_VARIANT              "XMSSMT-SHAKE256_40/4_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         4
#endif /* ifdef XMSSMT_SHAKE256_40_4_192 */

#ifdef XMSSMT_SHAKE256_40_8_192
#define XMSS_VARIANT              "XMSSMT-SHAKE256_40/8_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         8
#endif /* ifdef XMSSMT_SHAKE256_40_8_192 */

#ifdef XMSSMT_SHAKE256_60_3_192
#define XMSS_VARIANT              "XMSSMT-SHAKE256_60/3_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         3
#endif /* ifdef XMSSMT_SHAKE256_60_3_192 */

#ifdef XMSSMT_SHAKE256_60_6_192
#define XMSS_VARIANT              "XMSSMT-SHAKE256_60/6_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         6
#endif /* ifdef XMSSMT_SHAKE256_60_6_192 */

#ifdef XMSSMT_SHAKE256_60_12_192
#define XMSS_VARIANT              "XMSSMT-SHAKE256_60/12_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         12
#endif /* ifdef XMSSMT_SHAKE256_60_12_192 */

/***********************************
* XMSSMT SHA2 256
***********************************/
#ifdef XMSSMT_SHA2_20_2_256
#define XMSS_VARIANT              "XMSSMT-SHA2_20/2_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         2
#endif // ifdef XMSS_SHA2_20_2_256

#ifdef XMSSMT_SHA2_20_4_256
#define XMSS_VARIANT              "XMSSMT-SHA2_20/4_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         4
#endif /* ifdef XMSSMT_SHA2_20_4_256 */

#ifdef XMSSMT_SHA2_40_2_256
#define XMSS_VARIANT              "XMSSMT-SHA2_40/2_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         2
#endif /* ifdef XMSSMT_SHA2_40_2_256 */

#ifdef XMSSMT_SHA2_40_4_256
#define XMSS_VARIANT              "XMSSMT-SHA2_40/4_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         4
#endif /* ifdef XMSSMT_SHA2_40_4_256 */

#ifdef XMSSMT_SHA2_40_8_256
#define XMSS_VARIANT              "XMSSMT-SHA2_40/8_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         8
#endif /* ifdef XMSSMT_SHA2_40_8_256 */

#ifdef XMSSMT_SHA2_60_3_256
#define XMSS_VARIANT              "XMSSMT-SHA2_60/3_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         3
#endif /* ifdef XMSSMT_SHA2_60_6_256 */

#ifdef XMSSMT_SHA2_60_6_256
#define XMSS_VARIANT              "XMSSMT-SHA2_60/6_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         6
#endif /* ifdef XMSSMT_SHA2_60_6_256 */

#ifdef XMSSMT_SHA2_60_12_256
#define XMSS_VARIANT              "XMSSMT-SHA2_60/12_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         12
#endif /* ifdef XMSSMT_SHA2_60_12_256 */

/***********************************
* XMSSMT SHAKE 256
***********************************/
#ifdef XMSSMT_SHAKE256_20_2_256
#define XMSS_VARIANT              "XMSSMT-SHAKE256_20/2_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         2
#endif // ifdef XMSS_SHAKE256_20_2_256

#ifdef XMSSMT_SHAKE256_20_4_256
#define XMSS_VARIANT              "XMSSMT-SHAKE256_20/4_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         4
#endif /* ifdef XMSSMT_SHAKE256_20_4_256 */

#ifdef XMSSMT_SHAKE256_40_2_256
#define XMSS_VARIANT              "XMSSMT-SHAKE256_40/2_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         2
#endif /* ifdef XMSSMT_SHAKE256_40_2_256 */

#ifdef XMSSMT_SHAKE256_40_4_256
#define XMSS_VARIANT              "XMSSMT-SHAKE256_40/4_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         4
#endif /* ifdef XMSSMT_SHAKE256_40_4_256 */

#ifdef XMSSMT_SHAKE256_40_8_256
#define XMSS_VARIANT              "XMSSMT-SHAKE256_40/8_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               40
#define D                         8
#endif /* ifdef XMSSMT_SHAKE256_40_8_256 */

#ifdef XMSSMT_SHAKE256_60_3_256
#define XMSS_VARIANT              "XMSSMT-SHA2_60/3_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         3
#endif /* ifdef XMSSMT_SHA2_60_6_256 */

#ifdef XMSSMT_SHAKE256_60_6_256
#define XMSS_VARIANT              "XMSSMT-SHAKE256_60/6_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         6
#endif /* ifdef XMSSMT_SHAKE256_60_6_256 */

#ifdef XMSSMT_SHAKE256_60_12_256
#define XMSS_VARIANT              "XMSSMT-SHAKE256_60/12_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               60
#define D                         12
#endif /* ifdef XMSSMT_SHAKE256_60_12_256 */

/***********************************
* XMSS SHA256 192
***********************************/
#ifdef XMSS_SHA2_10_192
#define XMSS_VARIANT              "XMSS-SHA2_10_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               10
#define D                         1
#endif /* ifdef XMSS_SHA2_10_192 */

#ifdef XMSS_SHA2_16_192
#define XMSS_VARIANT              "XMSS-SHA2_16_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               16
#define D                         1
#endif /* ifdef XMSS_SHA2_16_192 */

#ifdef XMSS_SHA2_20_192
#define XMSS_VARIANT              "XMSS-SHA2_20_192"
#define FUNC                      XMSS_SHA2
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         1
#endif /* ifdef XMSS_SHA2_20_192 */

#ifdef XMSS_SHA2_10_256
#define XMSS_VARIANT              "XMSS-SHA2_10_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               10
#define D                         1
#endif /* ifdef XMSS_SHA2_10_256 */

#ifdef XMSS_SHA2_16_256
#define XMSS_VARIANT              "XMSS-SHA2_16_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               16
#define D                         1
#endif /* ifdef XMSS_SHA2_16_256 */

#ifdef XMSS_SHA2_20_256
#define XMSS_VARIANT              "XMSS-SHA2_20_256"
#define FUNC                      XMSS_SHA2
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         1
#endif /* ifdef XMSS_SHA2_20_256 */

#ifdef XMSS_SHAKE256_10_192
#define XMSS_VARIANT              "XMSS-SHAKE256_10_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               10
#define D                         1
#endif /* ifdef XMSS_SHAKE256_10_192 */

#ifdef XMSS_SHAKE256_16_192
#define XMSS_VARIANT              "XMSS-SHAKE256_16_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               16
#define D                         1
#endif /* ifdef XMSS_SHA2_10_256 */

#ifdef XMSS_SHAKE256_20_192
#define XMSS_VARIANT              "XMSS-SHAKE256_20_192"
#define FUNC                      XMSS_SHAKE256
#define N                         24
#define PADDING_LEN               4
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 48
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         1
#endif /* ifdef XMSS_SHA2_10_256 */

#ifdef XMSS_SHAKE256_10_256
#define XMSS_VARIANT              "XMSS-SHAKE256_10_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               10
#define D                         1
#endif /* ifdef XMSS_SHAKE256_10_256 */

#ifdef XMSS_SHAKE256_16_256
#define XMSS_VARIANT              "XMSS-SHAKE256_16_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               16
#define D                         1
#endif /* ifdef XMSS_SHAKE256_16_256 */

#ifdef XMSS_SHAKE256_20_256
#define XMSS_VARIANT              "XMSS-SHAKE256_20_256"
#define FUNC                      XMSS_SHAKE256
#define N                         32
#define PADDING_LEN               32
#define WOTS_W                    16
#define WOTS_LOG_W                4
#define WOTS_LEN1                 64
#define WOTS_LEN2                 3
#define FULL_HEIGHT               20
#define D                         1
#endif /* ifdef XMSS_SHAKE256_20_256 */

#define SIG_BYTES                 (INDEX_BYTES + N + D * WOTS_SIG_BYTES + FULL_HEIGHT * N)
#define PK_BYTES                  (2 * N)
#define WOTS_SIG_BYTES            (WOTS_LEN * N)
#define BDS_K                     0
#define TREE_HEIGHT               (FULL_HEIGHT / D)
#define WOTS_LEN                  (WOTS_LEN1 + WOTS_LEN2)

#ifdef XMSSMT
#define INDEX_BYTES               ((FULL_HEIGHT + 7) / 8)
#else  /* ifdef XMSSMT */
#define INDEX_BYTES               4
#endif /* ifdef XMSSMT */

#ifdef USING_BDS
#define SK_BYTES \
	(INDEX_BYTES + 4 * N \
	 + (2 * D - 1) * ( \
		 (TREE_HEIGHT + 1) * N \
		 + 4 \
		 + TREE_HEIGHT + 1 \
		 + TREE_HEIGHT * N \
		 + (TREE_HEIGHT >> 1) * N \
		 + (TREE_HEIGHT - BDS_K) * (7 + N) \
		 + ((1 << BDS_K) - BDS_K - 1) * N \
		 + 4 \
		 ) \
	 + (D - 1) * WOTS_SIG_BYTES)
#else  /* ifdef USING_BDS */
#define SK_BYTES                  (4 * N + INDEX_BYTES)

#endif /* ifdef USING_BDS */

#define SM_BYTES (SIG_BYTES + XMSS_MLEN)
