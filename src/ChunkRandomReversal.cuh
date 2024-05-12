#pragma once

#include "RNG.cuh"
#include <vector>

namespace {
	constexpr uint64_t MASK_48 = getBitmask(48);
	constexpr uint64_t M1 = 25214903917;
	constexpr uint64_t A1 = 11;
	constexpr uint64_t M2 = 205749139540585;
	constexpr uint64_t A2 = 277363943098;
	constexpr uint64_t M3 = 233752471717045;
	constexpr uint64_t A3 = 11718085204285;
	constexpr uint64_t M4 = 55986898099985;
	constexpr uint64_t A4 = 49720483695876;

	// Returns a population seed in 1.12-.
	uint64_t get_1_12_populationSeed(const uint64_t structureSeed, const uint64_t x, const uint64_t z) {
		uint64_t xoredStructureSeed = structureSeed ^ M1;
		uint32_t c1 = (((xoredStructureSeed * M1 + A1) & MASK_48)) >> 16;
		uint32_t c2 = (((xoredStructureSeed * M2 + A2) & MASK_48)) >> 16;
		uint32_t c3 = (((xoredStructureSeed * M3 + A3) & MASK_48)) >> 16;
		uint32_t c4 = (((xoredStructureSeed * M4 + A4) & MASK_48)) >> 16;
		uint64_t l1 = (static_cast<uint64_t>(c1) << 32) + static_cast<uint64_t>(static_cast<int32_t>(c2));
		uint64_t l2 = (static_cast<uint64_t>(c3) << 32) + static_cast<uint64_t>(static_cast<int32_t>(c4));

		uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(l1) / 2 * 2 + 1);
		uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(l2) / 2 * 2 + 1);
		return (x * a + z * b ^ structureSeed) & MASK_48;
	}

	// Returns the list of, and number of, possible offsets (which are XORed with a structure seed) in a 1.12- population seed.
	void get_1_12_offsets(uint64_t *offsets, int32_t *offsetsLength, const uint64_t x, const uint64_t z) {
		for (uint64_t i = 0; i < 3; i++) {
			for (uint64_t j = 0; j < 3; j++) {
				uint64_t offset = x * i + z * j;

				bool duplicate = false;
				for (int32_t k = 0; k < *offsetsLength; k++) {
					if (offsets[k] == offset) {
						duplicate = true;
						break;
					}
				}
				if (!duplicate) {
					offsets[*offsetsLength] = offset;
					(*offsetsLength)++;
				}
			}
		}
	}

	void reverse_1_12_populationSeed_recursiveFallback(const uint64_t offset, const uint64_t structureSeed, const int32_t numberOfKnownBitsInStructureSeed, const uint64_t populationSeed, const uint64_t x, const uint64_t z, std::vector<uint64_t> &out) {
		if ((((x * (((structureSeed ^ M1) * M2 + A2) >> 16) + z * (((structureSeed ^ M1) * M4 + A4) >> 16) + offset) ^ structureSeed ^ populationSeed) & getBitmask(numberOfKnownBitsInStructureSeed - 16)) != 0) return;
		if (numberOfKnownBitsInStructureSeed == 48) {
			if (get_1_12_populationSeed(structureSeed, x, z) == populationSeed) out.push_back(structureSeed);
			return;
		}
		reverse_1_12_populationSeed_recursiveFallback(offset, structureSeed, numberOfKnownBitsInStructureSeed + 1, populationSeed, x, z, out);
		reverse_1_12_populationSeed_recursiveFallback(offset, structureSeed + twoToThePowerOf(numberOfKnownBitsInStructureSeed), numberOfKnownBitsInStructureSeed + 1, populationSeed, x, z, out);
	}

	void reverse_1_12_populationSeed_fallback(const uint64_t populationSeed, const uint64_t x, const uint64_t z, std::vector<uint64_t> &out) {
		uint64_t offsets[9];
		int32_t offsetsLength = 0;
		get_1_12_offsets(offsets, &offsetsLength, x, z);

		for (int32_t k = 0; k < offsetsLength; k++) {
			uint64_t offset = offsets[k];
			for (uint64_t structure_seed_low = 0; structure_seed_low < twoToThePowerOf(16); structure_seed_low += 1) {
				reverse_1_12_populationSeed_recursiveFallback(offset, structure_seed_low, 16, populationSeed, x, z, out);
			}
		}
	}

	// Returns a population seed in 1.13+.
	uint64_t get_1_13_populationSeed(const uint64_t structureSeed, const uint64_t x, const uint64_t z) {
		uint64_t xoredStructureSeed = structureSeed ^ M1;
		uint32_t c1 = (((xoredStructureSeed * M1 + A1) & MASK_48)) >> 16;
		uint32_t c2 = (((xoredStructureSeed * M2 + A2) & MASK_48)) >> 16;
		uint32_t c3 = (((xoredStructureSeed * M3 + A3) & MASK_48)) >> 16;
		uint32_t c4 = (((xoredStructureSeed * M4 + A4) & MASK_48)) >> 16;
		uint64_t l1 = (static_cast<uint64_t>(c1) << 32) + static_cast<uint64_t>(static_cast<int32_t>(c2));
		uint64_t l2 = (static_cast<uint64_t>(c3) << 32) + static_cast<uint64_t>(static_cast<int32_t>(c4));

		uint64_t a = l1 | 1;
		uint64_t b = l2 | 1;
		return (x * a + z * b ^ structureSeed) & MASK_48;
	}

	// Returns the list of, and number of, possible offsets (which are XORed with a structure seed) in a 1.13+ population seed.
	void get_1_13_offsets(uint64_t *offsets, int32_t *offsetsLength, const uint64_t x, const uint64_t z) {
		for (uint64_t i = 0; i < 2; i++) {
			for (uint64_t j = 0; j < 2; j++) {
				uint64_t offset = x * i + z * j;

				bool duplicate = false;
				for (int32_t k = 0; k < *offsetsLength; k++) {
					if (offsets[k] == offset) {
						duplicate = true;
						break;
					}
				}

				if (!duplicate) {
					offsets[(*offsetsLength)++] = offset;
				}
			}
		}
	}

	void reverse_1_13_populationSeed_recursiveFallback(const uint64_t offset, const uint64_t structureSeed, const int32_t numberOfKnownBitsInStructureSeed, const uint64_t populationSeed, const uint64_t x, const uint64_t z, std::vector<uint64_t> &out) {
		if ((((x * (((structureSeed ^ M1) * M2 + A2) >> 16) + z * (((structureSeed ^ M1) * M4 + A4) >> 16) + offset) ^ structureSeed ^ populationSeed) & getBitmask(numberOfKnownBitsInStructureSeed - 16)) != 0) return;

		if (numberOfKnownBitsInStructureSeed == 48) {
			if (get_1_13_populationSeed(structureSeed, x, z) == populationSeed) out.push_back(structureSeed);
			return;
		}

		reverse_1_13_populationSeed_recursiveFallback(offset, structureSeed, numberOfKnownBitsInStructureSeed + 1, populationSeed, x, z, out);
		reverse_1_13_populationSeed_recursiveFallback(offset, structureSeed + twoToThePowerOf(numberOfKnownBitsInStructureSeed), numberOfKnownBitsInStructureSeed + 1, populationSeed, x, z, out);
	}

	void reverse_1_13_populationSeed_fallback(const uint64_t populationSeed, const uint64_t x, const uint64_t z, std::vector<uint64_t> &out) {
		uint64_t offsets[9];
		int32_t offsetsLength = 0;
		get_1_13_offsets(offsets, &offsetsLength, x, z);

		for (int32_t k = 0; k < offsetsLength; k++) {
			uint64_t offset = offsets[k];
			for (uint64_t structure_seed_low = 0; structure_seed_low < twoToThePowerOf(16); structure_seed_low += 1) {
				reverse_1_13_populationSeed_recursiveFallback(offset, structure_seed_low, 16, populationSeed, x, z, out);
			}
		}
	}

}

void reverse_1_12_populationSeed(const uint64_t populationSeed, const uint64_t x, const uint64_t z, std::vector<uint64_t> &out) {
	if (!x && !z) {
		out.push_back(populationSeed);
		return;
	}

	uint64_t constant_mult = x * M2 + z * M4;
	int32_t constant_mult_zeros = getNumberOfTrailingZeroes(constant_mult);
	if (constant_mult_zeros >= 16) {
		reverse_1_12_populationSeed_fallback(populationSeed, x, z, out);
		return;
	}
	uint64_t constant_mult_mod_inv = inverseModulo(constant_mult >> constant_mult_zeros);
	int32_t bits_per_iter = 16 - constant_mult_zeros;

	int32_t x_zeros = getNumberOfTrailingZeroes(x);
	int32_t z_zeros = getNumberOfTrailingZeroes(z);
	int32_t xz_zeros = getNumberOfTrailingZeroes(x | z);

	uint64_t offsets[9];
	int32_t offsetsLength = 0;
	get_1_12_offsets(offsets, &offsetsLength, x, z);

	uint64_t xored_structure_seed_low = (populationSeed ^ M1) & getBitmask(xz_zeros + 1) ^ ((x_zeros != z_zeros) << xz_zeros);
	for (; xored_structure_seed_low < twoToThePowerOf(16); xored_structure_seed_low += twoToThePowerOf(xz_zeros + 1)) {
		uint64_t addend_const_no_offset = x * ((xored_structure_seed_low * M2 + A2) >> 16) + z * ((xored_structure_seed_low * M4 + A4) >> 16);
		
		for (int32_t k = 0; k < offsetsLength; k++) {
			uint64_t offset = offsets[k];

			uint64_t addend_const = addend_const_no_offset + offset;
			uint64_t result_const = populationSeed ^ M1;

			bool invalid = false;
			uint64_t xoredStructureSeed = xored_structure_seed_low;
			int32_t xoredStructureSeedBits = 16;
			while (xoredStructureSeedBits < 48) {
				int32_t bits_left = 48 - xoredStructureSeedBits;
				int32_t bits_this_iter = bits_left - constant_mult_zeros;
				if (bits_this_iter > 16 - constant_mult_zeros) bits_this_iter = 16 - constant_mult_zeros;

				uint64_t addend = addend_const + (xoredStructureSeed >> 16) * constant_mult;
				uint64_t result = result_const ^ xoredStructureSeed;
				uint64_t mult_result = (result - addend) >> (xoredStructureSeedBits - 16);
				if (bits_this_iter <= 0) {
					if ((mult_result & getBitmask(bits_left)) != 0) {
						invalid = true;
						break;
					}

					break;
				}
				
				if ((mult_result & getBitmask(constant_mult_zeros)) != 0) {
					invalid = true;
					break;
				}
				mult_result >>= constant_mult_zeros;

				xoredStructureSeed += ((mult_result * constant_mult_mod_inv) & getBitmask(bits_this_iter)) << xoredStructureSeedBits;
				xoredStructureSeedBits += bits_this_iter;
			}

			if (!invalid && xoredStructureSeedBits == 48 - constant_mult_zeros) {
				for (uint64_t structureSeed = mask(xoredStructureSeed ^ M1, xoredStructureSeedBits); structureSeed < twoToThePowerOf(48); structureSeed += twoToThePowerOf(xoredStructureSeedBits)) {
					if (get_1_12_populationSeed(structureSeed, x, z) == populationSeed) {
						out.push_back(structureSeed);
					}
				}
			}
		}
	}
}

void reverse_1_13_populationSeed(const uint64_t populationSeed, const uint64_t x, const uint64_t z, std::vector<uint64_t> &out) {
	if (!x && !z) {
		out.push_back(populationSeed);
		return;
	}

	uint64_t constant_mult = x * M2 + z * M4;
	int32_t constant_mult_zeros = getNumberOfTrailingZeroes(constant_mult);
	if (constant_mult_zeros >= 16) {
		reverse_1_13_populationSeed_fallback(populationSeed, x, z, out);
		return;
	}
	uint64_t constant_mult_mod_inv = inverseModulo(constant_mult >> constant_mult_zeros);
	int32_t bits_per_iter = 16 - constant_mult_zeros;

	int32_t x_zeros = getNumberOfTrailingZeroes(x);
	int32_t z_zeros = getNumberOfTrailingZeroes(z);
	int32_t xz_zeros = getNumberOfTrailingZeroes(x | z);

	uint64_t offsets[4];
	int32_t offsetsLength = 0;
	get_1_13_offsets(offsets, &offsetsLength, x, z);

	uint64_t xored_structure_seed_low = (populationSeed ^ M1) & getBitmask(xz_zeros + 1) ^ ((x_zeros != z_zeros) << xz_zeros);
	for (; xored_structure_seed_low < twoToThePowerOf(16); xored_structure_seed_low += twoToThePowerOf(xz_zeros + 1)) {
		uint64_t addend_const_no_offset = x * ((xored_structure_seed_low * M2 + A2) >> 16) + z * ((xored_structure_seed_low * M4 + A4) >> 16);
		
		for (int32_t k = 0; k < offsetsLength; k++) {
			uint64_t offset = offsets[k];

			uint64_t addend_const = addend_const_no_offset + offset;
			uint64_t result_const = populationSeed ^ M1;

			bool invalid = false;
			uint64_t xoredStructureSeed = xored_structure_seed_low;
			int32_t xoredStructureSeedBits = 16;
			while (xoredStructureSeedBits < 48) {
				int32_t bits_left = 48 - xoredStructureSeedBits;
				int32_t bits_this_iter = bits_left - constant_mult_zeros;
				if (bits_this_iter > 16 - constant_mult_zeros) bits_this_iter = 16 - constant_mult_zeros;

				uint64_t addend = addend_const + (xoredStructureSeed >> 16) * constant_mult;
				uint64_t result = result_const ^ xoredStructureSeed;
				uint64_t mult_result = (result - addend) >> (xoredStructureSeedBits - 16);
				if (bits_this_iter <= 0) {
					if ((mult_result & getBitmask(bits_left)) != 0) {
						invalid = true;
						break;
					}

					break;
				}
				
				if ((mult_result & getBitmask(constant_mult_zeros)) != 0) {
					invalid = true;
					break;
				}
				mult_result >>= constant_mult_zeros;

				xoredStructureSeed += ((mult_result * constant_mult_mod_inv) & getBitmask(bits_this_iter)) << xoredStructureSeedBits;
				xoredStructureSeedBits += bits_this_iter;
			}

			if (!invalid && xoredStructureSeedBits == 48 - constant_mult_zeros) {
				for (uint64_t structureSeed = mask(xoredStructureSeed ^ M1, xoredStructureSeedBits); structureSeed < twoToThePowerOf(48); structureSeed += twoToThePowerOf(xoredStructureSeedBits)) {
					if (get_1_13_populationSeed(structureSeed, x, z) == populationSeed) {
						out.push_back(structureSeed);
					}
				}
			}
		}
	}
}