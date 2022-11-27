/* test for algorithmic parallelism: start */
#define USING_UNROLL
#define USING_PTX
#define USING_INTEGER
#define USING_COALESCED_ACCESS
#define USING_LOCAL_MEMORY
#define USING_PRE_COMP

#define USING_PARALLEL_WOTS_PKGEN
#define USING_PARALLEL_L_TREE

#define NOT_TEST_RATIO
/* test for algorithmic parallelism: finish */

/* test for multi-keypair data parallelism: start */
#define LARGE_SCHEME 1 // 1 or 2
#define USING_20_KEY
#define TEST_STREAM
// 1: 82 * 32
// 2: cuda core
// 3: maximum
#define USING_STREAM 2
/* test for multi-keypair data parallelism: finish */

#define SAME_CHECK

// #define USING_FIXED_SEEDS

#ifndef DEVICE_USED
#define DEVICE_USED 1
#endif /* ifndef DEVICE_USED */

#ifndef XMSS_SIGNATURES
#ifdef USING_BDS
#define XMSS_SIGNATURES 64
#else
#define XMSS_SIGNATURES 4
#endif
#endif // ifndef XMSS_SIGNATURES

// #define PRINT_ALL

#ifdef XMSSMT

#ifndef VARIANT
// #define XMSSMT_SHA2_20_2_192
// #define XMSSMT_SHA2_20_4_192
// #define XMSSMT_SHA2_40_2_192
// #define XMSSMT_SHA2_40_4_192
// #define XMSSMT_SHA2_40_8_192
// #define XMSSMT_SHA2_60_3_192
// #define XMSSMT_SHA2_60_6_192
// #define XMSSMT_SHA2_60_12_192

// #define XMSSMT_SHAKE256_20_2_192
// #define XMSSMT_SHAKE256_20_4_192
// #define XMSSMT_SHAKE256_40_2_192
// #define XMSSMT_SHAKE256_40_4_192
// #define XMSSMT_SHAKE256_40_8_192
// #define XMSSMT_SHAKE256_60_3_192
// #define XMSSMT_SHAKE256_60_6_192
// #define XMSSMT_SHAKE256_60_12_192

#define XMSSMT_SHA2_20_2_256
// #define XMSSMT_SHA2_20_4_256
// #define XMSSMT_SHA2_40_2_256
// #define XMSSMT_SHA2_40_4_256
// #define XMSSMT_SHA2_40_8_256
// #define XMSSMT_SHA2_60_3_256
// #define XMSSMT_SHA2_60_6_256
// #define XMSSMT_SHA2_60_12_256

// #define XMSSMT_SHAKE256_20_2_256
// #define XMSSMT_SHAKE256_20_4_256
// #define XMSSMT_SHAKE256_40_2_256
// #define XMSSMT_SHAKE256_40_4_256
// #define XMSSMT_SHAKE256_40_8_256
// #define XMSSMT_SHAKE256_60_3_256
// #define XMSSMT_SHAKE256_60_6_256
// #define XMSSMT_SHAKE256_60_12_256
#endif /* ifndef DXMSSMT_VARIANT */

#else // ifdef XMSSMT

#ifndef VARIANT
// #define XMSS_SHA2_10_192
// #define XMSS_SHA2_16_192
// #define XMSS_SHA2_20_192

// #define XMSS_SHA2_10_256
// #define XMSS_SHA2_16_256
#define XMSS_SHA2_20_256

// #define XMSS_SHAKE256_10_192
// #define XMSS_SHAKE256_16_192
// #define XMSS_SHAKE256_20_192

// #define XMSS_SHAKE256_10_256
// #define XMSS_SHAKE256_16_256
// #define XMSS_SHAKE256_20_256
#endif  // ifndef DXMSS_VARIANT

#endif  // ifdef XMSSMT
