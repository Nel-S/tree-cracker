#ifndef GENERATOR_H_
#define GENERATOR_H_

#include "biomenoise.cuh"

// generator flags
enum {
	LARGE_BIOMES            = 0x1,
	NO_BETA_OCEAN           = 0x2,
	FORCE_OCEAN_VARIANTS    = 0x4,
};

STRUCT(Generator) {
	int mc;
	int dim;
	uint32_t flags;
	uint64_t seed;
	uint64_t sha;

	union {
		struct { // MC 1.0 - 1.17
			LayerStack ls;
			Layer xlayer[5]; // buffer for custom entry layers @{1,4,16,64,256}
			Layer *entry;
		};
		struct { // MC 1.18
			BiomeNoise bn;
		};
		struct { // MC A1.2 - B1.7
			BiomeNoiseBeta bnb;
			//SurfaceNoiseBeta snb;
		};
	};
};


// #ifdef __cplusplus
// extern "C"
// {
// #endif

///=============================================================================
/// Biome Generation
///=============================================================================

/**
 * Calculates the buffer size (number of ints) required to generate a cuboidal
 * volume of size (sx, sy, sz). If 'sy' is zero the buffer is calculated for a
 * 2D plane (which is equivalent to sy=1 here).
 * The function allocCache() can be used to allocate the corresponding int
 * buffer using malloc().
 */
__host__ __device__ size_t getMinCacheSize(const Generator &g, int scale, int sx, int sy, int sz);
__host__ __device__ int *allocCache(const Generator &g, Range r);

/**
 * Gets the biome for a specified scaled position. Note that the scale should
 * be either 1 or 4, for block or biome coordinates respectively.
 * Returns none (-1) upon failure.
 */
__host__ __device__ int getBiomeAt(const Generator &g, int scale, int x, int y, int z);

/**
 * Returns the default layer that corresponds to the given scale.
 * Supported scales are {0, 1, 4, 16, 64, 256}. A scale of zero indicates the
 * custom entry layer 'g->entry'.
 * (Overworld, MC <= 1.17)
 */
__host__ __device__ const Layer *getLayerForScale(const Generator *g, int scale);


///=============================================================================
/// Layered Biome Generation (old interface up to 1.17)
///=============================================================================

__host__ __device__ void setupLayerStack(LayerStack *g, int mc, int largeBiomes);

/* Calculates the minimum size of the buffers required to generate an area of
 * dimensions 'sizeX' by 'sizeZ' at the specified layer.
 */
__host__ __device__ size_t getMinLayerCacheSize(const Layer *layer, int sizeX, int sizeZ);

/* Set up custom layers. */
__host__ __device__ Layer *setupLayer(Layer *l, mapfunc_t *map, int mc, int8_t zoom, int8_t edge, uint64_t saltbase, Layer *p, Layer *p2);

/* Generates the specified area using the current generator settings and stores
 * the biomeIDs in 'out'.
 * The biomeIDs will be indexed in the form: out[x + z*areaWidth]
 * It is recommended that 'out' is allocated using allocCache() for the correct
 * buffer size.
 */
__host__ __device__ int genArea(const Layer *layer, int *out, int areaX, int areaZ, int areaWidth, int areaHeight);

/**
 * Sets up a biome generator for a given MC version. The 'flags' can be used to
 * control LARGE_BIOMES or to FORCE_OCEAN_VARIANTS to enable ocean variants at
 * scales higher than normal.
 */
__host__ __device__ void setupGenerator(Generator &g, int mc, uint32_t flags) {
	g.mc = mc;
	g.dim = DIM_OVERWORLD;
	g.flags = flags;
	g.seed = 0;
	g.sha = 0;

	if (MC_B1_8 <= mc && mc <= MC_1_17) {
		// printf("AA ");
		setupLayerStack(&g.ls, mc, flags & LARGE_BIOMES);
		// printf("AB ");
		g.entry = NULL;
	}
	else if (MC_1_18 <= mc) initBiomeNoise(&g.bn, mc);
	else g.bnb.mc = mc;
}

__host__ __device__ void applySeed(Generator &g, uint64_t seed) {
	g.seed = seed;
	g.sha = 0;

	if (g.mc <= MC_B1_7) {
		setBetaBiomeSeed(&g.bnb, seed);
		//initSurfaceNoiseBeta(&g->snb, g->seed);
	}
	else if (g.mc <= MC_1_17) {
		printf("BA ");
		setLayerSeed(g.entry ? g.entry : g.ls.entry_1, seed);
		printf("BB ");
	}
	else /* if (g.mc >= MC_1_18) */ setBiomeSeed(&g.bn, seed, g.flags);
	if (MC_1_15 <= g.mc) {
		if (g.mc <= MC_1_17 && !g.entry) g.sha = g.ls.entry_1->startSalt;
		else g.sha = getVoronoiSHA(seed);
	}
}


__host__ __device__ size_t getMinCacheSize(const Generator &g, int scale, int sx, int sy, int sz) {
	if (sy == 0) sy = 1;
	size_t len = (size_t)sx * sz * sy;
	if (g.mc <= MC_B1_7 && scale <= 4 && !(g.flags & NO_BETA_OCEAN)) {
		int cellwidth = scale >> 1;
		int smin = (sx < sz ? sx : sz);
		int slen = ((smin >> (2 >> cellwidth)) + 1) * 2 + 1;
		len += slen * sizeof(SeaLevelColumnNoiseBeta);
	}
	else if (MC_B1_8 <= g.mc && g.mc <= MC_1_17 && g.dim == DIM_OVERWORLD) {
		// recursively check the layer stack for the max buffer
		const Layer *entry = getLayerForScale(&g, scale);
		if (!entry) {
			printf("getMinCacheSize(): failed to determine scaled entry\n");
			// exit(1);
			return 0;
		}
		size_t len2d = getMinLayerCacheSize(entry, sx, sz);
		len += len2d - sx*sz;
	}
	else if ((g.mc >= MC_1_18 || g.dim != DIM_OVERWORLD) && scale <= 1) {
		// allocate space for temporary copy of voronoi source
		sx = ((sx+3) >> 2) + 2;
		sy = ((sy+3) >> 2) + 2;
		sz = ((sz+3) >> 2) + 2;
		len += sx * sy * sz;
	}
	return len;
}

// __host__ __device__ size_t getMinCacheSize(const Generator *g) {
//     if (g->mc <= MC_B1_7 && !(g->flags & NO_BETA_OCEAN)) return 193;
//     else if (MC_B1_8 <= g->mc && g->mc <= MC_1_17 && g->dim == DIM_OVERWORLD) {
//         // recursively check the layer stack for the max buffer
//         if (!g->ls.entry_1) return 0;
//         int maxX = 1, maxZ = 1;
//         size_t bufsiz = 0;
//         getMaxArea(g->ls.entry_1, 1, 1, &maxX, &maxZ, &bufsiz);
//         return bufsiz + maxX * (size_t)maxZ;
//     }
//     else if (g->mc >= MC_1_18 || g->dim != DIM_OVERWORLD) return 28;
//     return 0;
// }

__host__ __device__ int *allocCache(const Generator &g, Range r) {
	size_t len = getMinCacheSize(g, r.scale, r.sx, r.sy, r.sz);
	// return (int*) calloc(len, sizeof(int));
	int *__temp;
	cudaMalloc(&__temp, len*sizeof(int));
	for (size_t __idx = 0; __idx < len; ++__idx) __temp[__idx] = 0;
	return __temp;
}

/**
 * Generates the biomes for a cuboidal scaled range given by 'r'.
 * (See description of Range for more detail.)
 *
 * The output is generated inside the cache. Upon success the biome ids can be
 * accessed by indexing as:
 *  cache[ y*r.sx*r.sz + z*r.sx + x ]
 * where (x,y,z) is an relative position inside the range cuboid.
 *
 * The required length of the cache can be determined with getMinCacheSize().
 *
 * The return value is zero upon success.
 */
__host__ __device__ int genBiomes(const Generator &g, int *cache, Range r) {
	int err = 1;
	int64_t i, k;

	if (MC_B1_8 <= g.mc && g.mc <= MC_1_17) {
		const Layer *entry = getLayerForScale(&g, r.scale);
		if (!entry) return -1;
		err = genArea(entry, cache, r.x, r.z, r.sx, r.sz);
		if (err) return err;
		for (k = 1; k < r.sy; k++) { // overworld has no vertical noise: expanding 2D into 3D
			for (i = 0; i < r.sx*r.sz; i++) cache[k*r.sx*r.sz + i] = cache[i];
		}
		return 0;
	} else if (MC_1_18 <= g.mc) return genBiomeNoiseScaled(&g.bn, cache, r, g.sha);
	else { // g->mc <= MC_B1_7
		if (g.flags & NO_BETA_OCEAN) err = genBiomeNoiseBetaScaled(&g.bnb, NULL, cache, r);
		else {
			SurfaceNoiseBeta snb;
			initSurfaceNoiseBeta(&snb, g.seed);
			err = genBiomeNoiseBetaScaled(&g.bnb, &snb, cache, r);
		}
		if (err) return err;
		for (k = 1; k < r.sy; k++) { // overworld has no vertical noise: expanding 2D into 3D
			for (i = 0; i < r.sx*r.sz; i++) cache[k*r.sx*r.sz + i] = cache[i];
		}
		return 0;
	}

	return err;
}

__host__ __device__ int getBiomeAt(const Generator &g, int scale, int x, int y, int z) {
	Range r = {scale, x, z, 1, 1, y, 1};
	int *ids = allocCache(g, r);
	int id = genBiomes(g, ids, r) ? none : ids[0];
	free(ids);
	return id;
}

__host__ __device__ const Layer *getLayerForScale(const Generator *g, int scale) {
	if (g->mc > MC_1_17)
		return NULL;
	switch (scale)
	{
	case 0:   return g->entry;
	case 1:   return g->ls.entry_1;
	case 4:   return g->ls.entry_4;
	case 16:  return g->ls.entry_16;
	case 64:  return g->ls.entry_64;
	case 256: return g->ls.entry_256;
	default:
		return NULL;
	}
}


__host__ __device__ Layer *setupLayer(Layer *l, mapfunc_t *map, int mc, int8_t zoom, int8_t edge, uint64_t saltbase, Layer *p, Layer *p2) {
	//Layer *l = g->layers + layerId;
	l->getMap = map;
	l->mc = mc;
	l->zoom = zoom;
	l->edge = edge;
	l->scale = 0;
	if (saltbase == 0 || saltbase == LAYER_INIT_SHA)
		l->layerSalt = saltbase;
	else
		l->layerSalt = getLayerSalt(saltbase);
	l->startSalt = 0;
	l->startSeed = 0;
	l->noise = NULL;
	l->data = NULL;
	l->p = p;
	l->p2 = p2;
	return l;
}

__host__ __device__ void setupScale(Layer &l, int scale) {
	l.scale = scale;
	if (l.p) setupScale(*l.p, scale * l.zoom);
	if (l.p2) setupScale(*l.p2, scale * l.zoom);
}

/* Initialize an instance of a layered generator. */
__host__ __device__ void setupLayerStack(LayerStack *g, int mc, int largeBiomes) {
	if (mc < MC_1_3) largeBiomes = 0;

	// cudaMemset(g, 0, sizeof(LayerStack));
	g->entry_1 = 0;
	g->entry_4 = 0;
	g->entry_16 = 0;
	g->entry_64 = 0;
	g->entry_256 = 0;

	Layer *p, *l = g->layers;
	mapfunc_t *map_land = 0;
	// L: layer
	// M: mapping function
	// V: minecraft version
	// Z: zoom
	// E: edge
	// S: salt base
	// P1: parent 1
	// P2: parent 2

	if (mc == MC_B1_8) {  
		//             L                   M               V   Z  E  S     P1 P2
		// NOTE: reusing slot for continent:4096, but scale is 1:8192
		map_land = mapLandB18;
		p = setupLayer(l+L_CONTINENT_4096, mapContinent,   mc, 1, 0, 1,    0, 0);
		p = setupLayer(l+L_ZOOM_4096,      mapZoomFuzzy,   mc, 2, 3, 2000, p, 0);
		p = setupLayer(l+L_LAND_4096,      map_land,       mc, 1, 2, 1,    p, 0);
		p = setupLayer(l+L_ZOOM_2048,      mapZoom,        mc, 2, 3, 2001, p, 0);
		p = setupLayer(l+L_LAND_2048,      map_land,       mc, 1, 2, 2,    p, 0);
		p = setupLayer(l+L_ZOOM_1024,      mapZoom,        mc, 2, 3, 2002, p, 0);
		p = setupLayer(l+L_LAND_1024_A,    map_land,       mc, 1, 2, 3,    p, 0);
		p = setupLayer(l+L_ZOOM_512,       mapZoom,        mc, 2, 3, 2003, p, 0);
		p = setupLayer(l+L_LAND_512,       map_land,       mc, 1, 2, 3,    p, 0);
		p = setupLayer(l+L_ZOOM_256,       mapZoom,        mc, 2, 3, 2004, p, 0);
		p = setupLayer(l+L_LAND_256,       map_land,       mc, 1, 2, 3,    p, 0);
		p = setupLayer(l+L_BIOME_256,      mapBiome,       mc, 1, 0, 200,  p, 0);
		p = setupLayer(l+L_ZOOM_128,       mapZoom,        mc, 2, 3, 1000, p, 0);
		p = setupLayer(l+L_ZOOM_64,        mapZoom,        mc, 2, 3, 1001, p, 0);
		// river noise layer chain, also used to determine where hills generate
		p = setupLayer(l+L_NOISE_256,      mapNoise,       mc, 1, 0, 100,
					   l+L_LAND_256, 0);
	} else if (mc <= MC_1_6) {
		//             L                   M               V   Z  E  S     P1 P2
		map_land = mapLand16;
		p = setupLayer(l+L_CONTINENT_4096, mapContinent,   mc, 1, 0, 1,    0, 0);
		p = setupLayer(l+L_ZOOM_2048,      mapZoomFuzzy,   mc, 2, 3, 2000, p, 0);
		p = setupLayer(l+L_LAND_2048,      map_land,       mc, 1, 2, 1,    p, 0);
		p = setupLayer(l+L_ZOOM_1024,      mapZoom,        mc, 2, 3, 2001, p, 0);
		p = setupLayer(l+L_LAND_1024_A,    map_land,       mc, 1, 2, 2,    p, 0);
		p = setupLayer(l+L_SNOW_1024,      mapSnow16,      mc, 1, 2, 2,    p, 0);
		p = setupLayer(l+L_ZOOM_512,       mapZoom,        mc, 2, 3, 2002, p, 0);
		p = setupLayer(l+L_LAND_512,       map_land,       mc, 1, 2, 3,    p, 0);
		p = setupLayer(l+L_ZOOM_256,       mapZoom,        mc, 2, 3, 2003, p, 0);
		p = setupLayer(l+L_LAND_256,       map_land,       mc, 1, 2, 4,    p, 0);
		p = setupLayer(l+L_MUSHROOM_256,   mapMushroom,    mc, 1, 2, 5,    p, 0);
		p = setupLayer(l+L_BIOME_256,      mapBiome,       mc, 1, 0, 200,  p, 0);
		p = setupLayer(l+L_ZOOM_128,       mapZoom,        mc, 2, 3, 1000, p, 0);
		p = setupLayer(l+L_ZOOM_64,        mapZoom,        mc, 2, 3, 1001, p, 0);
		// river noise layer chain, also used to determine where hills generate
		p = setupLayer(l+L_NOISE_256,      mapNoise,       mc, 1, 0, 100,
					   l+L_MUSHROOM_256, 0);
	} else {
		//             L                   M               V   Z  E  S     P1 P2
		map_land = mapLand;
		p = setupLayer(l+L_CONTINENT_4096, mapContinent,   mc, 1, 0, 1,    0, 0);
		p = setupLayer(l+L_ZOOM_2048,      mapZoomFuzzy,   mc, 2, 3, 2000, p, 0);
		p = setupLayer(l+L_LAND_2048,      map_land,       mc, 1, 2, 1,    p, 0);
		p = setupLayer(l+L_ZOOM_1024,      mapZoom,        mc, 2, 3, 2001, p, 0);
		p = setupLayer(l+L_LAND_1024_A,    map_land,       mc, 1, 2, 2,    p, 0);
		p = setupLayer(l+L_LAND_1024_B,    map_land,       mc, 1, 2, 50,   p, 0);
		p = setupLayer(l+L_LAND_1024_C,    map_land,       mc, 1, 2, 70,   p, 0);
		p = setupLayer(l+L_ISLAND_1024,    mapIsland,      mc, 1, 2, 2,    p, 0);
		p = setupLayer(l+L_SNOW_1024,      mapSnow,        mc, 1, 2, 2,    p, 0);
		p = setupLayer(l+L_LAND_1024_D,    map_land,       mc, 1, 2, 3,    p, 0);
		p = setupLayer(l+L_COOL_1024,      mapCool,        mc, 1, 2, 2,    p, 0);
		p = setupLayer(l+L_HEAT_1024,      mapHeat,        mc, 1, 2, 2,    p, 0);
		p = setupLayer(l+L_SPECIAL_1024,   mapSpecial,     mc, 1, 2, 3,    p, 0);
		p = setupLayer(l+L_ZOOM_512,       mapZoom,        mc, 2, 3, 2002, p, 0);
		p = setupLayer(l+L_ZOOM_256,       mapZoom,        mc, 2, 3, 2003, p, 0);
		p = setupLayer(l+L_LAND_256,       map_land,       mc, 1, 2, 4,    p, 0);
		p = setupLayer(l+L_MUSHROOM_256,   mapMushroom,    mc, 1, 2, 5,    p, 0);
		p = setupLayer(l+L_DEEP_OCEAN_256, mapDeepOcean,   mc, 1, 2, 4,    p, 0);
		p = setupLayer(l+L_BIOME_256,      mapBiome,       mc, 1, 0, 200,  p, 0);
		if (mc >= MC_1_14) p = setupLayer(l+L_BAMBOO_256, mapBamboo,      mc, 1, 0, 1001, p, 0);
		p = setupLayer(l+L_ZOOM_128,       mapZoom,        mc, 2, 3, 1000, p, 0);
		p = setupLayer(l+L_ZOOM_64,        mapZoom,        mc, 2, 3, 1001, p, 0);
		p = setupLayer(l+L_BIOME_EDGE_64,  mapBiomeEdge,   mc, 1, 2, 1000, p, 0);
		// river noise layer chain, also used to determine where hills generate
		p = setupLayer(l+L_RIVER_INIT_256, mapNoise,       mc, 1, 0, 100,
					   l+L_DEEP_OCEAN_256, 0);
	}

	if      (mc <= MC_1_0) {}
	else if (mc <= MC_1_12) {
		p = setupLayer(l+L_ZOOM_128_HILLS, mapZoom,        mc, 2, 3, 0,    p, 0);
		p = setupLayer(l+L_ZOOM_64_HILLS,  mapZoom,        mc, 2, 3, 0,    p, 0);
	} else { // if (mc >= MC_1_13)
		p = setupLayer(l+L_ZOOM_128_HILLS, mapZoom,        mc, 2, 3, 1000, p, 0);
		p = setupLayer(l+L_ZOOM_64_HILLS,  mapZoom,        mc, 2, 3, 1001, p, 0);
	}

	if (mc <= MC_1_0) {
		//             L                   M               V   Z  E  S     P1 P2
		p = setupLayer(l+L_ZOOM_32,        mapZoom,        mc, 2, 3, 1000,
					   l+L_ZOOM_64, 0);
		p = setupLayer(l+L_LAND_32,        map_land,       mc, 1, 2, 3,    p, 0);
		// NOTE: reusing slot for shore:16, but scale is 1:32
		p = setupLayer(l+L_SHORE_16,       mapShore,       mc, 1, 2, 1000, p, 0);
		p = setupLayer(l+L_ZOOM_16,        mapZoom,        mc, 2, 3, 1001, p, 0);
		p = setupLayer(l+L_ZOOM_8,         mapZoom,        mc, 2, 3, 1002, p, 0);
		p = setupLayer(l+L_ZOOM_4,         mapZoom,        mc, 2, 3, 1003, p, 0);

		p = setupLayer(l+L_SMOOTH_4,       mapSmooth,      mc, 1, 2, 1000, p, 0);

		// river layer chain
		p = setupLayer(l+L_ZOOM_128_RIVER, mapZoom,        mc, 2, 3, 1000,
					   l+L_NOISE_256, 0);
		p = setupLayer(l+L_ZOOM_64_RIVER,  mapZoom,        mc, 2, 3, 1001, p, 0);
		p = setupLayer(l+L_ZOOM_32_RIVER,  mapZoom,        mc, 2, 3, 1002, p, 0);
		p = setupLayer(l+L_ZOOM_16_RIVER,  mapZoom,        mc, 2, 3, 1003, p, 0);
		p = setupLayer(l+L_ZOOM_8_RIVER,   mapZoom,        mc, 2, 3, 1004, p, 0);
		p = setupLayer(l+L_ZOOM_4_RIVER,   mapZoom,        mc, 2, 3, 1005, p, 0);

		p = setupLayer(l+L_RIVER_4,        mapRiver,       mc, 1, 2, 1,    p, 0);
		p = setupLayer(l+L_SMOOTH_4_RIVER, mapSmooth,      mc, 1, 2, 1000, p, 0);
	} else if (mc <= MC_1_6) {
		//             L                   M               V   Z  E  S     P1 P2
		p = setupLayer(l+L_HILLS_64,       mapHills,       mc, 1, 2, 1000,
					   l+L_ZOOM_64, l+L_ZOOM_64_HILLS);

		p = setupLayer(l+L_ZOOM_32,        mapZoom,        mc, 2, 3, 1000, p, 0);
		p = setupLayer(l+L_LAND_32,        map_land,       mc, 1, 2, 3,    p, 0);
		p = setupLayer(l+L_ZOOM_16,        mapZoom,        mc, 2, 3, 1001, p, 0);
		p = setupLayer(l+L_SHORE_16,       mapShore,       mc, 1, 2, 1000, p, 0);
		p = setupLayer(l+L_SWAMP_RIVER_16, mapSwampRiver,  mc, 1, 0, 1000, p, 0);
		p = setupLayer(l+L_ZOOM_8,         mapZoom,        mc, 2, 3, 1002, p, 0);
		p = setupLayer(l+L_ZOOM_4,         mapZoom,        mc, 2, 3, 1003, p, 0);

		if (largeBiomes) {
			p = setupLayer(l+L_ZOOM_LARGE_A, mapZoom,      mc, 2, 3, 1004, p, 0);
			p = setupLayer(l+L_ZOOM_LARGE_B, mapZoom,      mc, 2, 3, 1005, p, 0);
		}

		p = setupLayer(l+L_SMOOTH_4,       mapSmooth,      mc, 1, 2, 1000, p, 0);

		// river layer chain
		p = setupLayer(l+L_ZOOM_128_RIVER, mapZoom,        mc, 2, 3, 1000,
					   l+L_NOISE_256, 0);
		p = setupLayer(l+L_ZOOM_64_RIVER,  mapZoom,        mc, 2, 3, 1001, p, 0);
		p = setupLayer(l+L_ZOOM_32_RIVER,  mapZoom,        mc, 2, 3, 1002, p, 0);
		p = setupLayer(l+L_ZOOM_16_RIVER,  mapZoom,        mc, 2, 3, 1003, p, 0);
		p = setupLayer(l+L_ZOOM_8_RIVER,   mapZoom,        mc, 2, 3, 1004, p, 0);
		p = setupLayer(l+L_ZOOM_4_RIVER,   mapZoom,        mc, 2, 3, 1005, p, 0);

		if (largeBiomes) {
			p = setupLayer(l+L_ZOOM_L_RIVER_A, mapZoom,    mc, 2, 3, 1006, p, 0);
			p = setupLayer(l+L_ZOOM_L_RIVER_B, mapZoom,    mc, 2, 3, 1007, p, 0);
		}

		p = setupLayer(l+L_RIVER_4,        mapRiver,       mc, 1, 2, 1,    p, 0);
		p = setupLayer(l+L_SMOOTH_4_RIVER, mapSmooth,      mc, 1, 2, 1000, p, 0);
	} else { // if (mc >= MC_1_7)
		//             L                   M               V   Z  E  S     P1 P2
		p = setupLayer(l+L_HILLS_64,       mapHills,       mc, 1, 2, 1000,
					   l+L_BIOME_EDGE_64, l+L_ZOOM_64_HILLS);

		p = setupLayer(l+L_SUNFLOWER_64,   mapSunflower,   mc, 1, 0, 1001, p, 0);
		p = setupLayer(l+L_ZOOM_32,        mapZoom,        mc, 2, 3, 1000, p, 0);
		p = setupLayer(l+L_LAND_32,        map_land,       mc, 1, 2, 3,    p, 0);
		p = setupLayer(l+L_ZOOM_16,        mapZoom,        mc, 2, 3, 1001, p, 0);
		p = setupLayer(l+L_SHORE_16,       mapShore,       mc, 1, 2, 1000, p, 0);
		p = setupLayer(l+L_ZOOM_8,         mapZoom,        mc, 2, 3, 1002, p, 0);
		p = setupLayer(l+L_ZOOM_4,         mapZoom,        mc, 2, 3, 1003, p, 0);

		if (largeBiomes) {
			p = setupLayer(l+L_ZOOM_LARGE_A, mapZoom,      mc, 2, 3, 1004, p, 0);
			p = setupLayer(l+L_ZOOM_LARGE_B, mapZoom,      mc, 2, 3, 1005, p, 0);
		}

		p = setupLayer(l+L_SMOOTH_4,       mapSmooth,      mc, 1, 2, 1000, p, 0);

		// river layer chain
		p = setupLayer(l+L_ZOOM_128_RIVER, mapZoom,        mc, 2, 3, 1000,
					   l+L_RIVER_INIT_256, 0);
		p = setupLayer(l+L_ZOOM_64_RIVER,  mapZoom,        mc, 2, 3, 1001, p, 0);
		p = setupLayer(l+L_ZOOM_32_RIVER,  mapZoom,        mc, 2, 3, 1000, p, 0);
		p = setupLayer(l+L_ZOOM_16_RIVER,  mapZoom,        mc, 2, 3, 1001, p, 0);
		p = setupLayer(l+L_ZOOM_8_RIVER,   mapZoom,        mc, 2, 3, 1002, p, 0);
		p = setupLayer(l+L_ZOOM_4_RIVER,   mapZoom,        mc, 2, 3, 1003, p, 0);

		if (largeBiomes && mc == MC_1_7) {
			p = setupLayer(l+L_ZOOM_L_RIVER_A, mapZoom,    mc, 2, 3, 1004, p, 0);
			p = setupLayer(l+L_ZOOM_L_RIVER_B, mapZoom,    mc, 2, 3, 1005, p, 0);
		}

		p = setupLayer(l+L_RIVER_4,        mapRiver,       mc, 1, 2, 1,    p, 0);
		p = setupLayer(l+L_SMOOTH_4_RIVER, mapSmooth,      mc, 1, 2, 1000, p, 0);
	}

	p = setupLayer(l+L_RIVER_MIX_4, mapRiverMix, mc, 1, 0, 100,
				   l+L_SMOOTH_4, l+L_SMOOTH_4_RIVER);


	if (mc <= MC_1_12) {
		p = setupLayer(l+L_VORONOI_1, mapVoronoi114, mc, 4, 3, 10, p, 0);
	} else {
		// ocean variants
		p = setupLayer(l+L_OCEAN_TEMP_256, mapOceanTemp,   mc, 1, 0, 2,    0, 0);
		p->noise = &g->oceanRnd;
		p = setupLayer(l+L_ZOOM_128_OCEAN, mapZoom,        mc, 2, 3, 2001, p, 0);
		p = setupLayer(l+L_ZOOM_64_OCEAN,  mapZoom,        mc, 2, 3, 2002, p, 0);
		p = setupLayer(l+L_ZOOM_32_OCEAN,  mapZoom,        mc, 2, 3, 2003, p, 0);
		p = setupLayer(l+L_ZOOM_16_OCEAN,  mapZoom,        mc, 2, 3, 2004, p, 0);
		p = setupLayer(l+L_ZOOM_8_OCEAN,   mapZoom,        mc, 2, 3, 2005, p, 0);
		p = setupLayer(l+L_ZOOM_4_OCEAN,   mapZoom,        mc, 2, 3, 2006, p, 0);
		p = setupLayer(l+L_OCEAN_MIX_4,    mapOceanMix,    mc, 1, 17, 100,
					   l+L_RIVER_MIX_4, l+L_ZOOM_4_OCEAN);

		if (mc <= MC_1_14) p = setupLayer(l+L_VORONOI_1, mapVoronoi114, mc, 4, 3, 10, p, 0);
		else p = setupLayer(l+L_VORONOI_1, mapVoronoi, mc, 4, 3, LAYER_INIT_SHA, p, 0);
	}
	g->entry_1 = p;
	g->entry_4 = l + (mc <= MC_1_12 ? L_RIVER_MIX_4 : L_OCEAN_MIX_4);
	if (largeBiomes) {
		g->entry_16 = l + L_ZOOM_4;
		g->entry_64 = l + (mc <= MC_1_6 ? L_SWAMP_RIVER_16 : L_SHORE_16);
		g->entry_256 = l + (mc <= MC_1_6 ? L_HILLS_64 : L_SUNFLOWER_64);
	} else if (MC_1_1 <= mc) {
		g->entry_16 = l + (mc <= MC_1_6 ? L_SWAMP_RIVER_16 : L_SHORE_16);
		g->entry_64 = l + (mc <= MC_1_6 ? L_HILLS_64 : L_SUNFLOWER_64);
		g->entry_256 = l + (mc <= MC_1_14 ? L_BIOME_256 : L_BAMBOO_256);
	} else {
		g->entry_16 = l + L_ZOOM_16;
		g->entry_64 = l + L_ZOOM_64;
		g->entry_256 = l + L_BIOME_256;
	}
	printf("AAA ");
	printf("(%d %d) ", g == NULL, g->entry_1 == NULL);
	if (g->entry_1) setupScale(*g->entry_1, 1); // PROBLEM #1
	printf("AAB ");
}


/* Recursively calculates the minimum buffer size required to generate an area
 * of the specified size from the current layer onwards.
 */
__host__ __device__ static void getMaxArea(const Layer *layer, int areaX, int areaZ, int *maxX, int *maxZ, size_t *siz) {
	if (layer == NULL)
		return;

	areaX += layer->edge;
	areaZ += layer->edge;

	// multi-layers and zoom-layers use a temporary copy of their parent area
	if (layer->p2 || layer->zoom != 1)
		*siz += areaX * areaZ;

	if (areaX > *maxX) *maxX = areaX;
	if (areaZ > *maxZ) *maxZ = areaZ;

	if (layer->zoom == 2)
	{
		areaX >>= 1;
		areaZ >>= 1;
	}
	else if (layer->zoom == 4)
	{
		areaX >>= 2;
		areaZ >>= 2;
	}

	getMaxArea(layer->p, areaX, areaZ, maxX, maxZ, siz);
	if (layer->p2)
		getMaxArea(layer->p2, areaX, areaZ, maxX, maxZ, siz);
}

__host__ __device__ size_t getMinLayerCacheSize(const Layer *layer, int sizeX, int sizeZ) {
	int maxX = sizeX, maxZ = sizeZ;
	size_t bufsiz = 0;
	getMaxArea(layer, sizeX, sizeZ, &maxX, &maxZ, &bufsiz);
	return bufsiz + maxX * (size_t)maxZ;
}

__host__ __device__ int genArea(const Layer *layer, int *out, int areaX, int areaZ, int areaWidth, int areaHeight) {
	// cudaMemset(out, 0, sizeof(*out)*areaWidth*areaHeight);
	for (size_t __index = 0; __index < areaWidth*areaHeight; ++__index) out[__index] = 0;
	return layer->getMap(layer, out, areaX, areaZ, areaWidth, areaHeight);
}

// #ifdef __cplusplus
// }
// #endif

#endif /* GENERATOR_H_ */