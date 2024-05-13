#ifndef __BIOMES_FILTER_CUH
#define __BIOMES_FILTER_CUH

#include "GenerateAndValidateData.cuh"
#include "cudabiomes/generator.cuh"

__managed__ uint64_t totalWorldseedsPerRun = 0;


__device__ constexpr int biomeToCubiomesBiome(const Biome biome) {
	switch (biome) {
		case Biome::Forest: return BiomeID::forest;
		case Biome::Birch_Forest: return BiomeID::birch_forest;
		case Biome::Taiga: return BiomeID::taiga;
		default:
			#if CUDA_IS_PRESENT
				return BiomeID::none;
			#else
				throw std::invalid_argument("ERROR: Unsupported version provided.\n");
			#endif
	}
}

__device__ constexpr int versionToCubiomesVersion(const Version version) {
	switch (version) {
		case Version::v1_6_4: return MCVersion::MC_1_6_4;
		case Version::v1_8_9: return MCVersion::MC_1_8_9;
		case Version::v1_12_2: return MCVersion::MC_1_12_2;
		case Version::v1_14_4: return MCVersion::MC_1_14_4;
		case Version::v1_16_1: return MCVersion::MC_1_16_1;
		case Version::v1_16_4: return MCVersion::MC_1_16_5;
		default:
			#if CUDA_IS_PRESENT
				return MCVersion::MC_UNDEF;
			#else
				throw std::invalid_argument("ERROR: Unsupported version provided.\n");
			#endif
	}
}

/* Reverses structure seeds (returned by ...) back to potential worldseeds.
   totalWorldseedsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in totalWorldseedsPerRun.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void biomeFilter() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t topBits = index % 65536;
	index /= 65536;
	if (index >= totalStructureSeedsPerRun) return;
	Generator g;
	int32_t ids[100]; //TODO: Calculate from getMinCacheSize()
	Version lastVersion;
	uint64_t resultIndex;
#else
void *biomeFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
	if (index >= totalStructureSeedsPerRun) return NULL;
	Generator g;
	int32_t ids[100]; //TODO: Calculate from getMinCacheSize()
	Version lastVersion;
	uint64_t resultIndex;
	for (uint64_t topBits = 0; topBits < 65536; ++topBits) {
#endif
		uint64_t seed = (topBits << 48) + filterInputs[index];
		for (uint32_t i = 0; i < POPULATION_CHUNKS_DATA.numberOfTreeChunks; ++i) {
			Version currentVersion = POPULATION_CHUNKS_DATA.treeChunks[i].version;
			if ((!topBits && !i) || currentVersion != lastVersion) {
				setupGenerator(&g, versionToCubiomesVersion(currentVersion), 0);
				applySeed(&g, DIM_OVERWORLD, seed);
				lastVersion = currentVersion;
			}
			int32_t populationChunkBlockX = getMinBlockCoordinate(POPULATION_CHUNKS_DATA.treeChunks[i].populationChunkX, currentVersion);
			int32_t populationChunkBlockZ = getMinBlockCoordinate(POPULATION_CHUNKS_DATA.treeChunks[i].populationChunkZ, currentVersion);
			// Range r = {1, populationChunkBlockX, populationChunkBlockZ, 16, 16, 64, 1};
			for (uint32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[i].numberOfTreePositions; ++j) {
				// getBiomeAt(&g, 1, populationChunkBlockX + POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkXOffset, 64, populationChunkBlockZ + POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkZOffset);
				Range r = {1, populationChunkBlockX + static_cast<int32_t>(POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkXOffset), populationChunkBlockZ + static_cast<int32_t>(POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkZOffset), 16, 16, 64, 1};
				if (genBiomes(&g, ids, r) || ids[0] != biomeToCubiomesBiome(POPULATION_CHUNKS_DATA.treeChunks[i].biome))
					#if CUDA_IS_PRESENT
						return;
					#else
						goto nextTopBits;
					#endif
			}
		}
		resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalWorldseedsPerRun), 1);
		if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
			#if (!CUDA_IS_PRESENT)
				NULL
			#endif
			;
		filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
		nextTopBits:
	}
	return NULL;
#endif
}

#endif