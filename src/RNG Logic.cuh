#ifndef __RNG_CUH
#define __RNG_CUH

#include "Base Logic.cuh"

struct LCG {
	static constexpr uint64_t MULTIPLIER = UINT64_C(0x5deece66d);
	static constexpr uint64_t ADDEND = UINT64_C(0xb);
	static constexpr uint64_t MASK = UINT64_C(0xffffffffffff);

	uint64_t multiplier, addend;

	__host__ __device__ constexpr explicit LCG(const uint64_t multiplier, const uint64_t addend) noexcept : multiplier(multiplier), addend(addend) {}

	__host__ __device__ constexpr void combine(const LCG &other) noexcept {
		uint64_t combined_multiplier = (this->multiplier * other.multiplier) & LCG::MASK;
		this->addend = (this->addend * other.multiplier + other.addend) & LCG::MASK;
		this->multiplier = combined_multiplier;
		// this->multiplier = (this->multiplier * other.multiplier) & LCG::MASK;
		// this->addend = (this->addend * other.multiplier + other.addend) & LCG::MASK;
	}

	__host__ __device__ [[nodiscard]] static constexpr LCG combine(const int64_t n) noexcept {
		uint64_t skip = static_cast<uint64_t>(n) & LCG::MASK;

		LCG lcg(1, 0);
		LCG lcg_pow(LCG::MULTIPLIER, LCG::ADDEND);

		for (uint64_t lowerBits = 0; lowerBits < 48; lowerBits++) {
			if (skip & twoToThePowerOf(lowerBits)) lcg.combine(lcg_pow);
			lcg_pow.combine(lcg_pow);
		}

		return lcg;
	}
};


struct Random {
	uint64_t seed;

	__host__ __device__ explicit Random() noexcept {}
	__host__ __device__ Random(const Random &other) noexcept : seed(other.seed) {}
	__host__ __device__ explicit Random(const uint64_t seed) noexcept : seed((seed ^ LCG::MULTIPLIER) & LCG::MASK) {}

	// Initialize a Random instance with an explicit given internal state.
	__host__ __device__ [[nodiscard]] static Random withSeed(const uint64_t state) noexcept {
		Random random;
		random.seed = state;
		return random;
	}

	// Initialize a Random instance using a given seed.
	__host__ __device__ void setSeed(const uint64_t seed) noexcept {
		this->seed = (seed ^ LCG::MULTIPLIER) & LCG::MASK;
	}
	// TODO: Convert bool to use Version
	__host__ __device__ uint64_t setPopulationSeed(const uint64_t worldSeed, const int32_t x, const int32_t z, const bool post1_12) noexcept {
		this->setSeed(worldSeed);
		uint64_t a, b;
		if (!post1_12) {
			a = static_cast<uint64_t>(static_cast<int64_t>(this->nextLong()) / 2 * 2 + 1);
			b = static_cast<uint64_t>(static_cast<int64_t>(this->nextLong()) / 2 * 2 + 1);
		} else {
			a = this->nextLong() | 1;
			b = this->nextLong() | 1;
		}
		uint64_t seed = (static_cast<uint64_t>(x) * a + static_cast<uint64_t>(z) * b) ^ worldSeed;
		this->setSeed(seed);
		return seed;
	}
	__host__ __device__ uint64_t setFeatureSeed(const uint64_t decorationSeed, const uint32_t index, const uint32_t step) noexcept {
		uint64_t seed = decorationSeed + static_cast<uint64_t>(index) + static_cast<uint64_t>(10000 * step);
		this->setSeed(seed);
		return seed;
	}
	__host__ __device__ uint64_t setLargeFeatureSeed(const uint64_t worldSeed, const int32_t chunkX, const int32_t chunkZ) noexcept {
		this->setSeed(worldSeed);
		uint64_t a = this->nextLong();
		uint64_t b = this->nextLong();
		uint64_t seed = (static_cast<uint64_t>(chunkX) * a ^ static_cast<uint64_t>(chunkZ) * b) ^ worldSeed;
		this->setSeed(seed);
		return seed;
	}
	__host__ __device__ uint64_t setLargeFeatureWithSalt(const uint64_t worldSeed, const int32_t regionX, const int32_t regionZ, const int32_t salt) noexcept {
		uint64_t seed = static_cast<uint64_t>(regionX) * UINT64_C(341873128712) + static_cast<uint64_t>(regionZ) * UINT64_C(132897987541) + worldSeed + static_cast<uint64_t>(salt);
		this->setSeed(seed);
		return seed;
	}

	template<int64_t N = 1>
	// Advances the Random instance by N states.
	__host__ __device__ Random &skip() noexcept {
		constexpr LCG lcg = LCG::combine(N);
		this->seed = (this->seed * lcg.multiplier + lcg.addend) & LCG::MASK;
		return *this;
	}
	__host__ __device__ Random &skip(const int64_t n) noexcept {
		// NelS: SURELY there must be a better way to do this.
		if (n & twoToThePowerOf(47)) this->skip<twoToThePowerOf(47)>();
		if (n & twoToThePowerOf(46)) this->skip<twoToThePowerOf(46)>();
		if (n & twoToThePowerOf(45)) this->skip<twoToThePowerOf(45)>();
		if (n & twoToThePowerOf(44)) this->skip<twoToThePowerOf(44)>();
		if (n & twoToThePowerOf(43)) this->skip<twoToThePowerOf(43)>();
		if (n & twoToThePowerOf(42)) this->skip<twoToThePowerOf(42)>();
		if (n & twoToThePowerOf(41)) this->skip<twoToThePowerOf(41)>();
		if (n & twoToThePowerOf(40)) this->skip<twoToThePowerOf(40)>();
		if (n & twoToThePowerOf(39)) this->skip<twoToThePowerOf(39)>();
		if (n & twoToThePowerOf(38)) this->skip<twoToThePowerOf(38)>();
		if (n & twoToThePowerOf(37)) this->skip<twoToThePowerOf(37)>();
		if (n & twoToThePowerOf(36)) this->skip<twoToThePowerOf(36)>();
		if (n & twoToThePowerOf(35)) this->skip<twoToThePowerOf(35)>();
		if (n & twoToThePowerOf(34)) this->skip<twoToThePowerOf(34)>();
		if (n & twoToThePowerOf(33)) this->skip<twoToThePowerOf(33)>();
		if (n & twoToThePowerOf(32)) this->skip<twoToThePowerOf(32)>();
		if (n & twoToThePowerOf(31)) this->skip<twoToThePowerOf(31)>();
		if (n & twoToThePowerOf(30)) this->skip<twoToThePowerOf(30)>();
		if (n & twoToThePowerOf(29)) this->skip<twoToThePowerOf(29)>();
		if (n & twoToThePowerOf(28)) this->skip<twoToThePowerOf(28)>();
		if (n & twoToThePowerOf(27)) this->skip<twoToThePowerOf(27)>();
		if (n & twoToThePowerOf(26)) this->skip<twoToThePowerOf(26)>();
		if (n & twoToThePowerOf(25)) this->skip<twoToThePowerOf(25)>();
		if (n & twoToThePowerOf(24)) this->skip<twoToThePowerOf(24)>();
		if (n & twoToThePowerOf(23)) this->skip<twoToThePowerOf(23)>();
		if (n & twoToThePowerOf(22)) this->skip<twoToThePowerOf(22)>();
		if (n & twoToThePowerOf(21)) this->skip<twoToThePowerOf(21)>();
		if (n & twoToThePowerOf(20)) this->skip<twoToThePowerOf(20)>();
		if (n & twoToThePowerOf(19)) this->skip<twoToThePowerOf(19)>();
		if (n & twoToThePowerOf(18)) this->skip<twoToThePowerOf(18)>();
		if (n & twoToThePowerOf(17)) this->skip<twoToThePowerOf(17)>();
		if (n & twoToThePowerOf(16)) this->skip<twoToThePowerOf(16)>();
		if (n & twoToThePowerOf(15)) this->skip<twoToThePowerOf(15)>();
		if (n & twoToThePowerOf(14)) this->skip<twoToThePowerOf(14)>();
		if (n & twoToThePowerOf(13)) this->skip<twoToThePowerOf(13)>();
		if (n & twoToThePowerOf(12)) this->skip<twoToThePowerOf(12)>();
		if (n & twoToThePowerOf(11)) this->skip<twoToThePowerOf(11)>();
		if (n & twoToThePowerOf(10)) this->skip<twoToThePowerOf(10)>();
		if (n & twoToThePowerOf( 9)) this->skip<twoToThePowerOf( 9)>();
		if (n & twoToThePowerOf( 8)) this->skip<twoToThePowerOf( 8)>();
		if (n & twoToThePowerOf( 7)) this->skip<twoToThePowerOf( 7)>();
		if (n & twoToThePowerOf( 6)) this->skip<twoToThePowerOf( 6)>();
		if (n & twoToThePowerOf( 5)) this->skip<twoToThePowerOf( 5)>();
		if (n & twoToThePowerOf( 4)) this->skip<twoToThePowerOf( 4)>();
		if (n & twoToThePowerOf( 3)) this->skip<twoToThePowerOf( 3)>();
		if (n & twoToThePowerOf( 2)) this->skip<twoToThePowerOf( 2)>();
		if (n & twoToThePowerOf( 1)) this->skip<twoToThePowerOf( 1)>();
		if (n & twoToThePowerOf( 0)) this->skip<twoToThePowerOf( 0)>();
		return *this;
	}

	template<int64_t N = 1>
	// Returns the next min([bits], 32)-bit integer, optionally skipping N states first.
	// The Random instance will be advanced once.
	__host__ __device__ uint32_t next(const uint32_t bits) noexcept {
		this->skip<N>();
		return static_cast<uint32_t>(this->seed >> (48 - bits));
	}

	template<int64_t N = 1>
	// Returns the next 32-bit integer, optionally skipping N states first.
	// The Random instance will be advanced once.
	__host__ __device__ uint32_t nextInt() noexcept {
		return this->next<N>(32);
	}
	template<int64_t N = 1>
	// Returns the next integer in the range [0, min(bound, 2^32 - 1)), optionally skipping N states first.
	// The Random instance will be advanced at least once, plus more with probability (2^31 mod bound)/2^31.
	__host__ __device__ uint32_t nextInt(const uint32_t bound) noexcept {
		uint32_t r = this->next<N>(31);
		uint32_t m = bound - 1;
		if (!(bound & m)) r = static_cast<uint32_t>((static_cast<uint64_t>(bound) * static_cast<uint64_t>(r)) >> 31);
		else {
			for (uint32_t u = r; static_cast<int32_t>(u - (r = u % bound) + m) < 0; u = this->next(31));
		}
		return r;
	}
	template<int64_t N = 1>
	// Returns the next integer in the range [0, min(bound, 2^32 - 1)), optionally skipping N states first.
	// The Random instance will always be advanced once, but the sequence of outputs will be (very slightly) biased towards smaller values.
	__host__ __device__ uint32_t nextIntFast(const uint32_t bound) noexcept {
		uint32_t r = this->next<N>(31);
		uint32_t m = bound - 1;
		if (!(bound & m)) r = static_cast<uint32_t>((static_cast<uint64_t>(bound) * static_cast<uint64_t>(r)) >> 31);
		else r %= bound;
		return r;
	}

	template<int64_t N = 1>
	// Returns the next 64-bit integer (though only 2^48 outputs are possible), optionally skipping N states first.
	// The Random instance will be advanced twice.
	__host__ __device__ uint64_t nextLong() noexcept {
		uint32_t a = this->next<N>(32);
		uint32_t b = this->next(32);
		return (static_cast<uint64_t>(a) << 32) + static_cast<uint64_t>(static_cast<int32_t>(b));
	}

	template<int64_t N = 1>
	// Returns the next Boolean value, optionally skipping N states first.
	// The Random instance will be advanced once.
	__host__ __device__ bool nextBoolean() noexcept {
		return !!this->next<N>(1);
	}

	template<int64_t N = 1>
	// Returns the next single-precision floating-point number, optionally skipping N states first.
	// The Random instance will be advanced once.
	__host__ __device__ float nextFloat() noexcept {
		return static_cast<float>(this->next<N>(24)) * 0x1.0p-24f;
	}

	template<int64_t N = 1>
	// Returns the next double-precision floating-point number, optionally skipping N states first.
	// The Random instance will be advanced twice.
	__host__ __device__ double nextDouble() noexcept {
		uint32_t a = this->next<N>(26);
		uint32_t b = this->next(27);
		return static_cast<double>((static_cast<uint64_t>(a) << 27) + static_cast<uint64_t>(b)) * 0x1.0p-53;
	}

	/* Returns whether a state could be generated as the result of a nextLong.
	   Adapted from Panda4994 (https://github.com/Panda4994/panda4994.github.io/blob/48526d35d3d38750102b9f360dff45a4bdbc50bd/seedinfo/js/Random.js#L16).*/
	__host__ __device__ [[nodiscard]] static constexpr bool isFromNextLong(const uint64_t state) noexcept {
		uint64_t secondNextLong = state & 0xffffffff;
		uint64_t firstNextLong = ((state >> 32) & 0xffffffff) + static_cast<uint64_t>(secondNextLong > 0x7fffffff);
		uint64_t upper32Bits = (firstNextLong << 16) * LCG::MULTIPLIER + LCG::ADDEND;
		for (uint64_t lower16Bits = 0; lower16Bits < 65536; ++lower16Bits) {
			if (((upper32Bits + lower16Bits*LCG::MULTIPLIER) & LCG::MASK) >> 16 == secondNextLong) return true;
		}
		return false;
	}

	/* Derives up to maxResults randomly-generatable worldseeds with a given structure seed and stores them in array. Returns their count.*/
	// TODO: Verify correctness
	// TODO: Find maximum number possible
	__device__ static constexpr uint32_t getRandomSeedsFromStructureSeed(uint64_t structureSeed, const size_t maxResults, uint64_t *array) noexcept {
		structureSeed &= LCG::MASK;
		uint32_t arrayLength = 0;
		uint64_t prevStateUpperBits = (structureSeed << 16) * 246154705703781 + 107048004364969; // Technically should be ANDed with LCG::MASK, but since prevState performs the AND anyways there's no point
		for (uint64_t lower16Bits = 0; lower16Bits < 65536; ++lower16Bits) {
			uint64_t prevState = (prevStateUpperBits + lower16Bits * 246154705703781) & LCG::MASK;
			if ((prevState >> 16) & 0xffff == structureSeed >> 32) {
				arrayLength = atomicAdd(reinterpret_cast<unsigned long long *>(array), 1);
				if (arrayLength >= maxResults) break;
				array[arrayLength] = ((prevState & 0xffff00000000) << 16) + structureSeed;
			}
		}
		return arrayLength;
	}
};

#endif