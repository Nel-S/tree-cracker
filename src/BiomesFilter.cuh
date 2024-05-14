#ifndef __BIOMES_FILTER_CUH
#define __BIOMES_FILTER_CUH

#include "GenerateAndValidateData.cuh"
#include "cudabiomes/generator.cuh"

__managed__ uint64_t totalWorldseedsPerRun = 0;
__managed__ int32_t largeBiomesFlag; // Yes, int, not bool.

// The returned value from Cubiomes' getMinCacheSize() under the specified version and large biomes state.
__device__ constexpr size_t getIDsArraySize(const Version version, const bool largeBiomes) {
	switch (version) {
		case Version::v1_6_4:
			return largeBiomes ? 2470 : 2326;
		case Version::v1_8_9:
		case Version::v1_12_2:
			return largeBiomes ? 4981 : 4909;
		case Version::v1_14_4:
		case Version::v1_16_1:
		case Version::v1_16_4:
			return largeBiomes ? 8044 : 8095; 
	}
	#if CUDA_IS_PRESENT
		return 0;
	#else
		throw std::invalid_argument("ERROR: Unsupported version.\n");
	#endif
}

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

__device__ constexpr bool biomeHasGrass(const int biome) {
	switch (biome) {
		case badlands:
		case badlands_plateau:
		case deep_warm_ocean:
		case desert:
		case desert_hills:
		case desert_lakes:
		case eroded_badlands:
		case ice_spikes:
		case modified_badlands_plateau:
		// TODO: Place pull request for Cubiomes to mark wooded badlands/modified wooded badlands as having grass
		// case modified_wooded_badlands_plateau:
		case mushroom_fields:
		case mushroom_field_shore:
		case none:
		case warm_ocean:
		// case wooded_badlands_plateau:
			return false;
		default:
			return true;
	}
}

/* Reverses structure seeds (returned by ...) back to potential worldseeds.
   totalWorldseedsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in totalWorldseedsPerRun.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void biomeFilter() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t topBits = index & 65535;
	index /= 65536;
#else
void *biomeFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif
	if (index >= totalStructureSeedsPerRun) return;
	Generator g;
	int32_t biomes[8095];
	Version lastVersion;
	uint64_t resultIndex;
#if (!CUDA_IS_PRESENT)
	for (uint64_t topBits = 0; topBits < 65536; ++topBits) {
#endif
		uint64_t seed = (topBits << 48) + filterInputs[index];
		for (uint32_t i = 0; i < POPULATION_CHUNKS_DATA.numberOfTreeChunks; ++i) {
			Version currentVersion = POPULATION_CHUNKS_DATA.treeChunks[i].version;
			if ((!topBits && !i) || currentVersion != lastVersion) {
				setupGenerator(&g, versionToCubiomesVersion(currentVersion), largeBiomesFlag);
				applySeed(&g, DIM_OVERWORLD, seed);
				lastVersion = currentVersion;
			}
			int32_t populationChunkBlockX = getMinBlockCoordinate(POPULATION_CHUNKS_DATA.treeChunks[i].populationChunkX, currentVersion);
			int32_t populationChunkBlockZ = getMinBlockCoordinate(POPULATION_CHUNKS_DATA.treeChunks[i].populationChunkZ, currentVersion);
			Range populationChunkRange = {1, populationChunkBlockX, populationChunkBlockZ, 16, 16, 64, 1};
			if (genBiomes(&g, biomes, populationChunkRange)) {
				#if CUDA_IS_PRESENT
					return;
				#else
					goto nextTopBits;
				#endif
			}

			// Ensure at least 1 block within the chunk is the desired biome
			int desiredBiome = static_cast<int>(POPULATION_CHUNKS_DATA.treeChunks[i].biome);
			bool containsBiome = false;
			for (uint32_t j = 0; j < 255; ++j) containsBiome |= (biomes[j] == desiredBiome);
			if (!containsBiome) {
				#if CUDA_IS_PRESENT
					return;
				#else
					goto nextTopBits;
				#endif
			}

			// For each tree, ensure its corresponding biome can generate dirt/grass to spawn a tree on
			#pragma unroll
			for (uint32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[i].numberOfTreePositions; ++j) {
				uint32_t correspondingIndex = 16*POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkZOffset + POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkXOffset;
				if (!biomeHasGrass(biomes[correspondingIndex])) {
					#if CUDA_IS_PRESENT
						return;
					#else
						goto nextTopBits;
					#endif
				}
			}
		}
		resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalWorldseedsPerRun), 1);
		if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
			#if CUDA_IS_PRESENT
				return;
			#else
				return NULL;
			#endif
		}
		filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
		nextTopBits:
	}
	return NULL;
#endif
}

#endif