#ifndef __BIOMES_FILTER_CUH
#define __BIOMES_FILTER_CUH

#include "Settings and Input Data Processing.cuh"
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

#endif