"""Near-duplicate detection with a difference hash (dHash) + union-find grouping.
Used to stop the same / almost-the-same X-ray landing in both train and test (data leakage)."""
import cv2
import numpy as np

_LUT = np.array([bin(i).count("1") for i in range(65536)], dtype=np.uint8)


def dhash(img, hash_size=16):
    """Binary gradient signature, returned as hash_size**2/64 uint64 words."""
    small = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.frombuffer(np.packbits(bits).tobytes(), dtype=np.uint64).copy()


def _popcount(x):
    s = np.uint64
    return (_LUT[(x & s(0xFFFF)).astype(np.int64)].astype(np.uint16)
            + _LUT[((x >> s(16)) & s(0xFFFF)).astype(np.int64)]
            + _LUT[((x >> s(32)) & s(0xFFFF)).astype(np.int64)]
            + _LUT[((x >> s(48)) & s(0xFFFF)).astype(np.int64)])


def near_duplicate_pairs(H, max_dist=12, chunk=256):
    """H: (n, words) uint64. Returns list of (i, j, hamming_distance) with i<j."""
    n, w = H.shape
    pairs = []
    for s in range(0, n, chunk):
        blk = H[s:s + chunk]
        d = np.zeros((len(blk), n), dtype=np.uint16)
        for k in range(w):
            d += _popcount(blk[:, None, k] ^ H[None, :, k])
        ii, jj = np.nonzero(d <= max_dist)
        for a, b in zip(ii + s, jj):
            if a < b:
                pairs.append((int(a), int(b), int(d[a - s, b])))
    return pairs


def group_pairs(n, pairs):
    """Union-find. Returns array of group ids (length n)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])
