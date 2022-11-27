#ifndef XMSS_RANDOMBYTES_H
#define XMSS_RANDOMBYTES_H

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Tries to read xlen bytes from a source of randomness, and writes them to x.
 */
void randombytes(unsigned char *x, unsigned long long xlen);

#ifdef __cplusplus
}
#endif

#endif
