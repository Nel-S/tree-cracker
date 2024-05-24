#ifndef __BIOMES_FILTER_CUH
#define __BIOMES_FILTER_CUH

#include "Settings and Input Data Processing.cuh"
#include "cudabiomes/generator.cuh"

__managed__ uint64_t totalWorldseedsPerRun = 0;
__managed__ int32_t largeBiomesFlag; // Yes, int, not bool (to make iterating easier).

// The returned value from Cubiomes' getMinCacheSize() under the specified version and large biomes state.
__device__ constexpr size_t getIDsArraySize(const Version version, const bool largeBiomes) {
	switch (version) {
		case static_cast<Version>(ExperimentalVersion::v1_6_4):
			return largeBiomes ? 2470 : 2326;
		case static_cast<Version>(ExperimentalVersion::v1_8_9):
		case static_cast<Version>(ExperimentalVersion::v1_12_2):
			return largeBiomes ? 4981 : 4909;
		case Version::v1_14_4:
		case Version::v1_16_1:
		case Version::v1_16_4:
			return largeBiomes ? 8044 : 8095;
		default: THROW_EXCEPTION(0, "ERROR: Unsupported version provided.\n")
	}
}

__device__ constexpr int biomeToCubiomesBiome(const Biome biome) {
	switch (biome) {
		case Biome::Forest: return BiomeID::forest;
		case Biome::Birch_Forest: return BiomeID::birch_forest;
		case static_cast<Biome>(ExperimentalBiome::Taiga): return BiomeID::taiga;
		default: THROW_EXCEPTION(BiomeID::none, "ERROR: Unsupported biome provided.\n")
	}
}

__device__ constexpr int versionToCubiomesVersion(const Version version) {
	switch (version) {
		case static_cast<Version>(ExperimentalVersion::v1_6_4): return MCVersion::MC_1_6_4;
		case static_cast<Version>(ExperimentalVersion::v1_8_9): return MCVersion::MC_1_8_9;
		case static_cast<Version>(ExperimentalVersion::v1_12_2): return MCVersion::MC_1_12_2;
		case Version::v1_14_4: return MCVersion::MC_1_14_4;
		case Version::v1_16_1: return MCVersion::MC_1_16_1;
		case Version::v1_16_4: return MCVersion::MC_1_16_5;
		default: THROW_EXCEPTION(MCVersion::MC_UNDEF, "ERROR: Unsupported version provided.\n")
	}
}


// Returns the approximate chance of a specified biome generating under a specified version (see [../Biome Frequency Statistics]).
__host__ __device__ constexpr double getBiomeRarity(const Biome biome, const Version version) {
	switch (version) {
		case static_cast<Version>(ExperimentalVersion::v1_6_4):
			switch (biome) {
				case Biome::Birch_Forest: return 0.;
				case Biome::Forest: return 0.0883953;
				case static_cast<Biome>(ExperimentalBiome::Taiga): return 0.0843568;
				default: THROW_EXCEPTION(0., "ERROR: Unsupported biome provided.\n")
			}
		case static_cast<Version>(ExperimentalVersion::v1_8_9):
		case static_cast<Version>(ExperimentalVersion::v1_12_2):
			switch (biome) {
				case Biome::Birch_Forest: return 0.0269248;
				case Biome::Forest: return 0.0904358;
				case static_cast<Biome>(ExperimentalBiome::Taiga): return 0.0385804;
				default: THROW_EXCEPTION(0., "ERROR: Unsupported biome provided.\n")
			}
		case Version::v1_14_4:
			switch (biome) {
				case Biome::Birch_Forest: return 0.0269201;
				case Biome::Forest: return 0.0904048;
				case static_cast<Biome>(ExperimentalBiome::Taiga): return 0.0385741;
				default: THROW_EXCEPTION(0., "ERROR: Unsupported biome provided.\n")
			}
		case Version::v1_16_1:
		case Version::v1_16_4:
			switch (biome) {
				case Biome::Birch_Forest: return 0.0259779;
				case Biome::Forest: return 0.0870516;
				case static_cast<Biome>(ExperimentalBiome::Taiga): return 0.0372210;
				default: THROW_EXCEPTION(0., "ERROR: Unsupported biome provided.\n")
			}
		default: THROW_EXCEPTION(0., "ERROR: Unsupported version provided.\n")
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

const char *getNotesAboutSeed(const uint64_t seed, const bool largeBiomesFlag) noexcept {
	bool textSeedFlag = (INT32_MIN <= static_cast<int64_t>(seed) && static_cast<int64_t>(seed) <= INT32_MAX);
	bool randomSeedFlag = Random::isFromNextLong(seed);

	std::string output = "";
	if (textSeedFlag) {
		if (!output.size()) output += "\t(";
		else output += ", ";
		output += "Text seed";
	}
	if (randomSeedFlag) {
		if (!output.size()) output += "\t(";
		else output += ", ";
		output += "Random seed";
	}
	if (largeBiomesFlag) {
		if (!output.size()) output += "\t(";
		else output += ", ";
		output += "Large Biomes seed";
	}
	return (output + ")").c_str();
}

#endif