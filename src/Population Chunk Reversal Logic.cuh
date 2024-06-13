#ifndef __CHUNK_RANDOM_REVERSAL_CUH
#define __CHUNK_RANDOM_REVERSAL_CUH

#include "Settings and Input Data Processing.cuh"

__managed__ uint64_t totalStructureSeedsThisWorkerSet = 0;
uint64_t totalStructureSeedsThisRun = 0;
#define POPULATION_REVERSAL_INPUT  filterStorageB
#define POPULATION_REVERSAL_OUTPUT filterStorageA

constexpr uint64_t FORWARD_2_MULTIPLIER = UINT64_C(205749139540585);
constexpr uint64_t FORWARD_2_ADDEND = UINT64_C(277363943098);
constexpr uint64_t FORWARD_4_MULTIPLIER = UINT64_C(55986898099985);
constexpr uint64_t FORWARD_4_ADDEND = UINT64_C(49720483695876);

// Returns a population seed.
// TODO: Combine with Random in RNG.cuh
__device__ [[nodiscard]] uint64_t getPopulationSeed(const uint64_t structureSeed, const int32_t x, const int32_t z, const Version version) {
	Random random(structureSeed);
	uint64_t a, b;
	if (version <= static_cast<Version>(ExperimentalVersion::v1_12_2)) {
		a = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
		b = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
	} else {
		a = random.nextLong() | 1;
		b = random.nextLong() | 1;
	}
	return (static_cast<uint64_t>(x) * a + static_cast<uint64_t>(z) * b ^ structureSeed) & LCG::MASK;
}

/* Steps the */
static constexpr const int32_t __TEMP_MAX_CALLS = 2000;

__host__ __device__ void stepBackMinPopulationCalls(Random &random, const Biome biome, const Version version) {
	// 1.13+ uses tree salts (getTreeSalt()) instead
	if (static_cast<Version>(ExperimentalVersion::v1_12_2) < version) return;
	switch (version) {
		case static_cast<Version>(ExperimentalVersion::v1_6_4): // At most 3759
			random.skip<-3759 + __TEMP_MAX_CALLS/2>();
			break;
		case static_cast<Version>(ExperimentalVersion::v1_8_9): // At most 5056
			random.skip<-5056 + __TEMP_MAX_CALLS/2>();
			break;
		case static_cast<Version>(ExperimentalVersion::v1_12_2):
			random.skip<-0>();
			break;
		default: THROW_EXCEPTION(/* None */, "ERROR: Unsupported version provided.\n");
	}
}

__host__ __device__ [[nodiscard]] uint32_t getPopulationCallsRange(const Biome biome, const Version version) {
	if (static_cast<Version>(ExperimentalVersion::v1_12_2) < version) return 1;
	switch (version) {
		case static_cast<Version>(ExperimentalVersion::v1_6_4):
		case static_cast<Version>(ExperimentalVersion::v1_8_9): // At least 20
		case static_cast<Version>(ExperimentalVersion::v1_12_2):
			// Seems to have been arbitrarily picked
			return __TEMP_MAX_CALLS;
		default: THROW_EXCEPTION(UINT32_MAX, "ERROR: Unsupported version provided.\n");
	}
}

/* Returns the list of, and number of, possible offsets related to population seeds.
   These are possible displacements that arise from the population seed formula turning both nextLongs into odd integers. 1.12- rounds the nextLongs, meaning there are up to 3^2 = 9 possible combinations that could have occurred internally; 1.13+ sets the last bit to 1, meaning there are up to 2^2 = 4 possible combinations.*/
__device__ void getInternalPopulationOffsets(uint64_t *offsets, uint32_t *offsetsLength, const int32_t x, const int32_t z, const Version version) {
	for (uint64_t i = 0; i < 2 + static_cast<uint64_t>(version <= static_cast<Version>(ExperimentalVersion::v1_12_2)); ++i) {
		for (uint64_t j = 0; j < 2 + static_cast<uint64_t>(version <= static_cast<Version>(ExperimentalVersion::v1_12_2)); ++j) {
			uint64_t offset = static_cast<uint64_t>(x) * i + static_cast<uint64_t>(z) * j;

			for (uint32_t k = 0; k < *offsetsLength; ++k) {
				if (offsets[k] == offset) goto skipOffset; // NelS: Yes, I used a goto to break out of multiple nested loops. Sue me.
			}
			
			offsets[*offsetsLength] = offset;
			++(*offsetsLength);

			skipOffset: continue;
		}
	}
}

/* Recursively derives the structure seeds from the population seeds.
   If successful, the results are placed in POPULATION_REVERSAL_OUTPUT[] and totalStructureSeedsThisWorkerSet is incremented.*/
__device__ void reversePopulationSeedRecursiveFallback(const uint64_t offset, const uint64_t partialStructureSeed, const int32_t numberOfKnownBitsInStructureSeed, const uint64_t populationSeed, const int32_t x, const int32_t z, const Version version) {
	// First tests if the last (numberOfKnownBitsInStructureSeed - 16) bits of the structure seed, when placed into the population seed formula and combined with the specified offset, produce the last (numberOfKnownBitsInStructureSeed - 16) bits of the true population seed. If not, quit.
	if (getLowestBitsOf((static_cast<uint64_t>(x) * (((partialStructureSeed ^ LCG::MULTIPLIER) * FORWARD_2_MULTIPLIER + FORWARD_2_ADDEND) >> 16) + static_cast<uint64_t>(z) * (((partialStructureSeed ^ LCG::MULTIPLIER) * FORWARD_4_MULTIPLIER + FORWARD_4_ADDEND) >> 16) + offset) ^ partialStructureSeed ^ populationSeed, numberOfKnownBitsInStructureSeed - 16)) return;
	// Otherwise, if the full structure seed has been determined, test if it satisfies the full population seed formula; if so add it to the list, then quit regardless
	if (numberOfKnownBitsInStructureSeed == 48) {
		if (getPopulationSeed(partialStructureSeed, x, z, version) != populationSeed) return;
		uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalStructureSeedsThisWorkerSet), 1);
		if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return;
		// if (ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1) POPULATION_REVERSAL_OUTPUT[resultIndex] = partialStructureSeed;
		// else POPULATION_REVERSAL_OUTPUT[resultIndex] = {resultIndex, populationSeed};
		POPULATION_REVERSAL_OUTPUT[resultIndex] = partialStructureSeed;
		return;
	}
	// Otherwise, try adding one more bit to the partial structure seeds (one attempt for 0, one attempt for 1) and repeat
	reversePopulationSeedRecursiveFallback(offset, partialStructureSeed, numberOfKnownBitsInStructureSeed + 1, populationSeed, x, z, version);
	reversePopulationSeedRecursiveFallback(offset, partialStructureSeed + twoToThePowerOf(numberOfKnownBitsInStructureSeed), numberOfKnownBitsInStructureSeed + 1, populationSeed, x, z, version);
}

__device__ void reversePopulationSeed(const uint64_t populationSeed, const int32_t x, const int32_t z, const Version version) {
	// If x = z = 0, the structure seed is just the population seed
	if (!x && !z) {
		uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalStructureSeedsThisWorkerSet), 1);
		if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return;
		// if (ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1) POPULATION_REVERSAL_OUTPUT[resultIndex] = populationSeed;
		// else POPULATION_REVERSAL_OUTPUT[resultIndex] = {resultIndex, populationSeed};
		POPULATION_REVERSAL_OUTPUT[resultIndex] = populationSeed;
		return;
	}

	uint64_t offsets[9]; // NelS: Ideally should be length 9 for 1.12- and 4 for 1.13+, but there doesn't seem to be an "easy" way to specify that
	uint32_t offsetsLength = 0;
	getInternalPopulationOffsets(offsets, &offsetsLength, x, z, version);

	uint64_t constant_mult = static_cast<uint64_t>(x)*FORWARD_2_MULTIPLIER + static_cast<uint64_t>(z)*FORWARD_4_MULTIPLIER;
	int32_t constant_mult_zeros = getNumberOfTrailingZeroes(constant_mult);
	if (constant_mult_zeros >= 16) {
		for (uint32_t k = 0; k < offsetsLength; ++k) {
			for (uint64_t structureSeedLowerBits = 0; structureSeedLowerBits < 65536; ++structureSeedLowerBits) reversePopulationSeedRecursiveFallback(offsets[k], structureSeedLowerBits, 16, populationSeed, x, z, version);
		}
		return;
	}

	uint64_t constant_mult_mod_inv = inverseModulo(constant_mult >> constant_mult_zeros);
	int32_t xz_zeros = getNumberOfTrailingZeroes(x | z);

	for (uint64_t xored_structure_seed_low = getLowestBitsOf(populationSeed ^ LCG::MULTIPLIER, xz_zeros + 1) ^ ((getNumberOfTrailingZeroes(x) != getNumberOfTrailingZeroes(z)) << xz_zeros); xored_structure_seed_low < 65536; xored_structure_seed_low += twoToThePowerOf(xz_zeros + 1)) {
		uint64_t addendConstantWithoutOffset = static_cast<uint64_t>(x)*((xored_structure_seed_low * FORWARD_2_MULTIPLIER + FORWARD_2_ADDEND) >> 16) + static_cast<uint64_t>(z)*((xored_structure_seed_low * FORWARD_4_MULTIPLIER + FORWARD_4_ADDEND) >> 16);

		for (uint32_t k = 0; k < offsetsLength; ++k) {
			uint64_t addendConstant = addendConstantWithoutOffset + offsets[k];
			uint64_t result_const = populationSeed ^ LCG::MULTIPLIER;

			bool isInvalid = false;
			uint64_t xoredStructureSeed = xored_structure_seed_low;
			int32_t xoredStructureSeedBits = 16;
			while (xoredStructureSeedBits < 48) {
				int32_t bits_left = 48 - xoredStructureSeedBits;
				int32_t bits_this_iter = constexprMin(bits_left, 16) - constant_mult_zeros;

				uint64_t mult_result = ((result_const ^ xoredStructureSeed) - addendConstant - (xoredStructureSeed >> 16) * constant_mult) >> (xoredStructureSeedBits - 16);
				if (bits_this_iter <= 0) {
					if (getLowestBitsOf(mult_result, bits_left)) isInvalid = true;
					break;
				}
				if (getLowestBitsOf(mult_result, constant_mult_zeros)) {
					isInvalid = true;
					break;
				}
				mult_result >>= constant_mult_zeros;

				xoredStructureSeed += getLowestBitsOf(mult_result * constant_mult_mod_inv, bits_this_iter) << xoredStructureSeedBits;
				xoredStructureSeedBits += bits_this_iter;
			}
			if (isInvalid || xoredStructureSeedBits != 48 - constant_mult_zeros) continue;

			for (uint64_t structureSeed = getLowestBitsOf(xoredStructureSeed ^ LCG::MULTIPLIER, xoredStructureSeedBits); structureSeed <= LCG::MASK; structureSeed += twoToThePowerOf(xoredStructureSeedBits)) {
				if (getPopulationSeed(structureSeed, x, z, version) != populationSeed) continue;
				uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalStructureSeedsThisWorkerSet), 1);
				if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return;
				// if (ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1) POPULATION_REVERSAL_OUTPUT[resultIndex] = structureSeed;
				// else POPULATION_REVERSAL_OUTPUT[resultIndex] = {resultIndex, populationSeed};
				POPULATION_REVERSAL_OUTPUT[resultIndex] = structureSeed;
			}
		}
	}
}

#endif