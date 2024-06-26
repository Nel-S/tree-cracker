#ifndef NOISE_H_
#define NOISE_H_

#include "cbRng.cuh"
#include <cmath>
#include <cstdio>

STRUCT(PerlinNoise) {
	uint8_t d[257];
	uint8_t h2;
	double a, b, c, amplitude, lacunarity, d2, t2;
};

STRUCT(OctaveNoise) {
	int octcnt;
	PerlinNoise *octaves;
};

STRUCT(DoublePerlinNoise) {
	double amplitude;
	OctaveNoise octA;
	OctaveNoise octB;
};

// #ifdef __cplusplus
// extern "C"
// {
// #endif

// grad()
#if 0
__host__ __device__ static double indexedLerp(int idx, double d1, double d2, double d3)
{
	const double cEdgeX[] = { 1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0,
							  0.0, 0.0, 0.0, 0.0, 1.0, 0.0,-1.0, 0.0 };
	const double cEdgeY[] = { 1.0, 1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0,
							  1.0,-1.0, 1.0,-1.0, 1.0,-1.0, 1.0,-1.0 };
	const double cEdgeZ[] = { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,-1.0,-1.0,
							  1.0, 1.0,-1.0,-1.0, 0.0, 1.0, 0.0,-1.0 };

	idx &= 0xf;
	return cEdgeX[idx] * d1 + cEdgeY[idx] * d2 + cEdgeZ[idx] * d3;
}
#else
__host__ __device__ ATTR(hot, const)
static inline double indexedLerp(uint8_t idx, double a, double b, double c) {
	switch (idx & 0xf) {
		case 0: case 12: return  a + b;
		case 1: case 14: return -a + b;
		case 2:  return  a - b;
		case 3:  return -a - b;
		case 4:  return  a + c;
		case 5:  return -a + c;
		case 6:  return  a - c;
		case 7:  return -a - c;
		case 8:  return  b + c;
		case 9: case 13: return -b + c;
		case 10: return  b - c;
		case 11: case 15: return -b - c;
	}
	return 0;
}
#endif

__host__ __device__ void perlinInit(PerlinNoise *noise, Random &random) {
	int i = 0;
	//memset(noise, 0, sizeof(*noise));
	noise->a = random.nextDouble() * 256.0;
	noise->b = random.nextDouble() * 256.0;
	noise->c = random.nextDouble() * 256.0;
	noise->amplitude = 1.0;
	noise->lacunarity = 1.0;

	uint8_t *idx = noise->d;
	for (i = 0; i < 256; i++) idx[i] = i;
	for (i = 0; i < 256; i++) {
		int j = random.nextInt(256 - i) + i;
		uint8_t n = idx[i];
		idx[i] = idx[j];
		idx[j] = n;
	}
	idx[256] = idx[0];
	double i2 = floor(noise->b);
	double d2 = noise->b - i2;
	noise->h2 = (int) i2;
	noise->d2 = d2;
	noise->t2 = d2*d2*d2 * (d2 * (d2*6.0-15.0) + 10.0);
}

__host__ __device__ void perlinInit(PerlinNoise *noise, Xoroshiro &random) {
	int i = 0;
	//memset(noise, 0, sizeof(*noise));
	noise->a = random.nextDouble() * 256.0;
	noise->b = random.nextDouble() * 256.0;
	noise->c = random.nextDouble() * 256.0;
	noise->amplitude = 1.0;
	noise->lacunarity = 1.0;

	uint8_t *idx = noise->d;
	for (i = 0; i < 256; i++) idx[i] = i;
	for (i = 0; i < 256; i++) {
		int j = random.nextInt(256 - i) + i;
		uint8_t n = idx[i];
		idx[i] = idx[j];
		idx[j] = n;
	}
	idx[256] = idx[0];
	double i2 = floor(noise->b);
	double d2 = noise->b - i2;
	noise->h2 = (int) i2;
	noise->d2 = d2;
	noise->t2 = d2*d2*d2 * (d2 * (d2*6.0-15.0) + 10.0);
}

__host__ __device__ double samplePerlin(const PerlinNoise *noise, double d1, double d2, double d3,
		double yamp, double ymin)
{
	uint8_t h1, h2, h3;
	double t1, t2, t3;

	if (d2 == 0.0) {
		d2 = noise->d2;
		h2 = noise->h2;
		t2 = noise->t2;
	} else {
		d2 += noise->b;
		double i2 = floor(d2);
		d2 -= i2;
		h2 = (int) i2;
		t2 = d2*d2*d2 * (d2 * (d2*6.0-15.0) + 10.0);
	}

	d1 += noise->a;
	d3 += noise->c;

	double i1 = floor(d1);
	double i3 = floor(d3);
	d1 -= i1;
	d3 -= i3;

	h1 = (int) i1;
	h3 = (int) i3;

	t1 = d1*d1*d1 * (d1 * (d1*6.0-15.0) + 10.0);
	t3 = d3*d3*d3 * (d3 * (d3*6.0-15.0) + 10.0);

	if (yamp)
	{
		double yclamp = ymin < d2 ? ymin : d2;
		d2 -= floor(yclamp / yamp) * yamp;
	}

	const uint8_t *idx = noise->d;

#if 1
	// try to promote optimizations that can utilize the {xh, xl} registers
	typedef struct vec2 { uint8_t a, b; } vec2;

	vec2 v1 = { idx[h1], idx[h1+1] };
	v1.a += h2;
	v1.b += h2;

	vec2 v2 = { idx[v1.a], idx[v1.a+1] };
	vec2 v3 = { idx[v1.b], idx[v1.b+1] };
	v2.a += h3;
	v2.b += h3;
	v3.a += h3;
	v3.b += h3;

	vec2 v4 = { idx[v2.a], idx[v2.a+1] };
	vec2 v5 = { idx[v2.b], idx[v2.b+1] };
	vec2 v6 = { idx[v3.a], idx[v3.a+1] };
	vec2 v7 = { idx[v3.b], idx[v3.b+1] };

	double l1 = indexedLerp(v4.a, d1,   d2,   d3);
	double l5 = indexedLerp(v4.b, d1,   d2,   d3-1);
	double l2 = indexedLerp(v6.a, d1-1, d2,   d3);
	double l6 = indexedLerp(v6.b, d1-1, d2,   d3-1);
	double l3 = indexedLerp(v5.a, d1,   d2-1, d3);
	double l7 = indexedLerp(v5.b, d1,   d2-1, d3-1);
	double l4 = indexedLerp(v7.a, d1-1, d2-1, d3);
	double l8 = indexedLerp(v7.b, d1-1, d2-1, d3-1);
#else
	uint8_t a1 = idx[h1]   + h2;
	uint8_t b1 = idx[h1+1] + h2;

	uint8_t a2 = idx[a1]   + h3;
	uint8_t b2 = idx[b1]   + h3;
	uint8_t a3 = idx[a1+1] + h3;
	uint8_t b3 = idx[b1+1] + h3;

	double l1 = indexedLerp(idx[a2],   d1,   d2,   d3);
	double l2 = indexedLerp(idx[b2],   d1-1, d2,   d3);
	double l3 = indexedLerp(idx[a3],   d1,   d2-1, d3);
	double l4 = indexedLerp(idx[b3],   d1-1, d2-1, d3);
	double l5 = indexedLerp(idx[a2+1], d1,   d2,   d3-1);
	double l6 = indexedLerp(idx[b2+1], d1-1, d2,   d3-1);
	double l7 = indexedLerp(idx[a3+1], d1,   d2-1, d3-1);
	double l8 = indexedLerp(idx[b3+1], d1-1, d2-1, d3-1);
#endif

	l1 = lerp(t1, l1, l2);
	l3 = lerp(t1, l3, l4);
	l5 = lerp(t1, l5, l6);
	l7 = lerp(t1, l7, l8);

	l1 = lerp(t2, l1, l3);
	l5 = lerp(t2, l5, l7);

	return lerp(t3, l1, l5);
}

__host__ __device__ static void samplePerlinBeta17Terrain(const PerlinNoise *noise, double *v, double d1, double d3, double yLacAmp) {
	int genFlag = -1;
	double l1 = 0;
	double l3 = 0;
	double l5 = 0;
	double l7 = 0;

	d1 += noise->a;
	d3 += noise->c;
	const uint8_t *idx = noise->d;
	int i1 = (int) floor(d1);
	int i3 = (int) floor(d3);
	d1 -= i1;
	d3 -= i3;
	double t1 = d1*d1*d1 * (d1 * (d1*6.0-15.0) + 10.0);
	double t3 = d3*d3*d3 * (d3 * (d3*6.0-15.0) + 10.0);

	i1 &= 0xff;
	i3 &= 0xff;

	double d2;
	int i2, yi, yic = 0, gfCopy = 0;
	for (yi = 0; yi <= 7; yi++)
	{
		d2 = yi*noise->lacunarity*yLacAmp+noise->b;
		i2 = ((int) floor(d2)) & 0xff;
		if (yi == 0 || i2 != genFlag)
		{
			yic = yi;
			gfCopy = genFlag;
			genFlag = i2;
		}
	}
	genFlag = gfCopy;

	double t2;
	for (yi = yic; yi <= 8; yi++)
	{
		d2 = yi*noise->lacunarity*yLacAmp+noise->b;
		i2 = (int) floor(d2);
		d2 -= i2;
		t2 = d2*d2*d2 * (d2 * (d2*6.0-15.0) + 10.0);

		i2 &= 0xff;

		if (yi == 0 || i2 != genFlag)
		{
			genFlag = i2;
			int a1 = idx[i1]   + i2;
			int b1 = idx[i1+1] + i2;

			int a2 = idx[a1]   + i3;
			int a3 = idx[a1+1] + i3;
			int b2 = idx[b1]   + i3;
			int b3 = idx[b1+1] + i3;

			double m1 = indexedLerp(idx[a2],   d1,   d2,   d3);
			double l2 = indexedLerp(idx[b2],   d1-1, d2,   d3);
			double m3 = indexedLerp(idx[a3],   d1,   d2-1, d3);
			double l4 = indexedLerp(idx[b3],   d1-1, d2-1, d3);
			double m5 = indexedLerp(idx[a2+1], d1,   d2,   d3-1);
			double l6 = indexedLerp(idx[b2+1], d1-1, d2,   d3-1);
			double m7 = indexedLerp(idx[a3+1], d1,   d2-1, d3-1);
			double l8 = indexedLerp(idx[b3+1], d1-1, d2-1, d3-1);

			l1 = lerp(t1, m1, l2);
			l3 = lerp(t1, m3, l4);
			l5 = lerp(t1, m5, l6);
			l7 = lerp(t1, m7, l8);
		}

		if (yi >= 7)
		{
			double n1 = lerp(t2, l1, l3);
			double n5 = lerp(t2, l5, l7);

			v[yi-7] += lerp(t3, n1, n5) * noise->amplitude;
		}
	}
}

__host__ __device__ static double simplexGrad(int idx, double x, double y, double z, double d) {
	double con = d - x*x - y*y - z*z;
	if (con < 0) return 0;
	con *= con;
	return con * con * indexedLerp(idx, x, y, z);
}

__host__ __device__ double sampleSimplex2D(const PerlinNoise *noise, double x, double y) {
	const double SKEW = 0.5 * (sqrtf(3) - 1.0);
	const double UNSKEW = (3.0 - sqrtf(3)) / 6.0;

	double hf = (x + y) * SKEW;
	int hx = (int)floor(x + hf);
	int hz = (int)floor(y + hf);
	double mhxz = (hx + hz) * UNSKEW;
	double x0 = x - (hx - mhxz);
	double y0 = y - (hz - mhxz);
	int offx = (x0 > y0);
	int offz = !offx;
	double x1 = x0 - offx + UNSKEW;
	double y1 = y0 - offz + UNSKEW;
	double x2 = x0 - 1.0 + 2.0 * UNSKEW;
	double y2 = y0 - 1.0 + 2.0 * UNSKEW;
	int gi0 = noise->d[0xff & (hz)];
	int gi1 = noise->d[0xff & (hz + offz)];
	int gi2 = noise->d[0xff & (hz + 1)];
	gi0 = noise->d[0xff & (gi0 + hx)];
	gi1 = noise->d[0xff & (gi1 + hx + offx)];
	gi2 = noise->d[0xff & (gi2 + hx + 1)];
	double t = 0;
	t += simplexGrad(gi0 % 12, x0, y0, 0.0, 0.5);
	t += simplexGrad(gi1 % 12, x1, y1, 0.0, 0.5);
	t += simplexGrad(gi2 % 12, x2, y2, 0.0, 0.5);
	return 70.0 * t;
}

__host__ __device__ void octaveInitBeta(OctaveNoise *noise, Random &random, PerlinNoise *octaves, int octcnt, double lac, double lacMul, double persist, double persistMul) {
	int i;
	for (i = 0; i < octcnt; i++) {
		perlinInit(&octaves[i], random);
		octaves[i].amplitude = persist;
		octaves[i].lacunarity = lac;
		persist *= persistMul;
		lac *= lacMul;
	}
	noise->octaves = octaves;
	noise->octcnt = octcnt;
}

__host__ __device__ int octaveInit(OctaveNoise *noise, Xoroshiro &random, PerlinNoise *octaves, const double *amplitudes, int omin, int len, int nmax) {
	static const uint64_t md5_octave_n[][2] = {
		{0xb198de63a8012672, 0x7b84cad43ef7b5a8}, // md5 "octave_-12"
		{0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8}, // md5 "octave_-11"
		{0x36d326eed40efeb2, 0x5be9ce18223c636a}, // md5 "octave_-10"
		{0x082fe255f8be6631, 0x4e96119e22dedc81}, // md5 "octave_-9"
		{0x0ef68ec68504005e, 0x48b6bf93a2789640}, // md5 "octave_-8"
		{0xf11268128982754f, 0x257a1d670430b0aa}, // md5 "octave_-7"
		{0xe51c98ce7d1de664, 0x5f9478a733040c45}, // md5 "octave_-6"
		{0x6d7b49e7e429850a, 0x2e3063c622a24777}, // md5 "octave_-5"
		{0xbd90d5377ba1b762, 0xc07317d419a7548d}, // md5 "octave_-4"
		{0x53d39c6752dac858, 0xbcd1c5a80ab65b3e}, // md5 "octave_-3"
		{0xb4a24d7a84e7677b, 0x023ff9668e89b5c4}, // md5 "octave_-2"
		{0xdffa22b534c5f608, 0xb9b67517d3665ca9}, // md5 "octave_-1"
		{0xd50708086cef4d7c, 0x6e1651ecc7f43309}, // md5 "octave_0"
	};
	static const double lacuna_ini[] = { // -omin = 3..12
		1, .5, .25, 1./8, 1./16, 1./32, 1./64, 1./128, 1./256, 1./512, 1./1024,
		1./2048, 1./4096,
	};
	static const double persist_ini[] = { // len = 4..9
		0, 1, 2./3, 4./7, 8./15, 16./31, 32./63, 64./127, 128./255, 256./511,
	};
	double lacuna = lacuna_ini[-omin];
	double persist = persist_ini[len];
	uint64_t xlo = random.nextLong();
	uint64_t xhi = random.nextLong();
	int i = 0, n = 0;

	for (; i < len && n != nmax; i++, lacuna *= 2.0, persist *= 0.5) {
		if (!amplitudes[i]) continue;
		Xoroshiro pxr = Xoroshiro::withState(xlo ^ md5_octave_n[12 + omin + i][0], xhi ^ md5_octave_n[12 + omin + i][1]);
		perlinInit(&octaves[n], pxr);
		octaves[n].amplitude = amplitudes[i] * persist;
		octaves[n].lacunarity = lacuna;
		n++;
	}

	noise->octaves = octaves;
	noise->octcnt = n;
	return n;
}


__host__ __device__ double sampleOctaveAmp(const OctaveNoise *noise, double x, double y, double z, double yamp, double ymin, int ydefault) {
	double v = 0;
	int i;
	for (i = 0; i < noise->octcnt; i++) {
		PerlinNoise *p = noise->octaves + i;
		double lf = p->lacunarity;
		double ax = x * lf;
		double ay = ydefault ? -p->b : y * lf;
		double az = z * lf;
		double pv = samplePerlin(p, ax, ay, az, yamp * lf, ymin * lf);
		v += p->amplitude * pv;
	}
	return v;
}

__host__ __device__ double sampleOctave(const OctaveNoise *noise, double x, double y, double z) {
	double v = 0;
	int i;
	for (i = 0; i < noise->octcnt; i++) {
		PerlinNoise *p = noise->octaves + i;
		double lf = p->lacunarity;
		double ax = x * lf;
		double ay = y * lf;
		double az = z * lf;
		double pv = samplePerlin(p, ax, ay, az, 0, 0);
		v += p->amplitude * pv;
	}
	return v;
}

__host__ __device__ double sampleOctaveBeta17Biome(const OctaveNoise *noise, double x, double z) {
	double v = 0;
	int i;
	for (i = 0; i < noise->octcnt; i++)
	{
		PerlinNoise *p = noise->octaves + i;
		double lf = p->lacunarity;
		double ax = x * lf + p->a;
		double az = z * lf + p->b;
		double pv = sampleSimplex2D(p, ax, az);
		v += p->amplitude * pv;
	}
	return v;
}

__host__ __device__ void sampleOctaveBeta17Terrain(const OctaveNoise *noise, double *v,
		double x, double z, int yLacFlag, double lacmin)
{
	v[0] = 0.0;
	v[1] = 0.0;
	int i;
	for (i = 0; i < noise->octcnt; i++)
	{
		PerlinNoise *p = noise->octaves + i;
		double lf = p->lacunarity;
		if (lacmin && lf > lacmin)
			continue;
		double ax = x * lf;
		double az = z * lf;
		samplePerlinBeta17Terrain(p, v, ax, az, yLacFlag ? 0.5 : 1.0);
	}
}

/**
 * Sets up a DoublePerlinNoise generator (MC 1.18+).
 * @noise:      Object to be initialized
 * @xr:         Xoroshiro random object
 * @octaves:    Octaves buffer, size has to be 2x (# non-zeros in amplitudes)
 * @amplitudes: Octave amplitude, needs at least one non-zero
 * @omin:       First octave
 * @len:        Length of amplitudes array
 * @nmax:       Number of octaves available in buffer (can be <=0 to ignore)
 * @return Number of octaves used (see octaves buffer size).
 */
__host__ __device__ int doublePerlinInit(DoublePerlinNoise *noise, Xoroshiro &random, PerlinNoise *octaves, const double *amplitudes, int omin, int len, int nmax) {
	int i, n = 0, na = -1, nb = -1;
	if (nmax > 0) {
		na = (nmax + 1) >> 1;
		nb = nmax - na;
	}
	n += octaveInit(&noise->octA, random, octaves+n, amplitudes, omin, len, na);
	n += octaveInit(&noise->octB, random, octaves+n, amplitudes, omin, len, nb);

	// trim amplitudes of zero
	for (i = len-1; i >= 0 && amplitudes[i] == 0.0; i--)
		len--;
	for (i = 0; amplitudes[i] == 0.0; i++)
		len--;
	static const double amp_ini[] = { // (5 ./ 3) * len / (len + 1), len = 2..9
		0, 5./6, 10./9, 15./12, 20./15, 25./18, 30./21, 35./24, 40./27, 45./30,
	};
	noise->amplitude = amp_ini[len];
	return n;
}

__host__ __device__ double sampleDoublePerlin(const DoublePerlinNoise *noise,
		double x, double y, double z)
{
	const double f = 337.0 / 331.0;
	double v = 0;

	v += sampleOctave(&noise->octA, x, y, z);
	v += sampleOctave(&noise->octB, x*f, y*f, z*f);

	return v * noise->amplitude;
}


// #ifdef __cplusplus
// }
// #endif

#endif /* NOISE_H_ */