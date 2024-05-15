#ifndef __TREES_CUH
#define __TREES_CUH

#include "RNG Logic.cuh"
#include "../Allowed Values for Settings.cuh"

// Given a block coordinate along one axis, and the version, returns the corresponding population chunk coordinate along that axis.
__device__ constexpr int32_t getPopulationChunkCoordinate(const int32_t blockCoordinate, const Version version) noexcept {
	return (blockCoordinate - 8*static_cast<int32_t>(version <= Version::v1_12_2)) >> 4;
}

// Given a block coordinate along one axis, and the version, returns the block's offset along that axis within the corresponding population chunk.
__device__ constexpr uint32_t getOffsetWithinPopulationChunk(const int32_t blockCoordinate, const Version version) noexcept {
	return (blockCoordinate - 8*static_cast<int32_t>(version <= Version::v1_12_2)) & 15;
}

// Given a population chunk coordinate along one axis, and the version, returns its corresponding minimum block coordinate along that axis.
__host__ __device__ constexpr int32_t getMinBlockCoordinate(const int32_t populationChunkCoordinate, const Version version) noexcept {
	return 16*(populationChunkCoordinate + 8*static_cast<int32_t>(version <= Version::v1_12_2));
}

// Given a population chunk coordinate along one axis, and the version, returns its corresponding maximum block coordinate along that axis.
__host__ __device__ constexpr int32_t getMaxBlockCoordinate(const int32_t populationChunkCoordinate, const Version version) noexcept {
	return getMinBlockCoordinate(populationChunkCoordinate, version) + 15;
}


// Set containing a combination of tree types.
struct SetOfTreeTypes {
	uint32_t treeTypeFlags;

	__device__ constexpr SetOfTreeTypes() noexcept : treeTypeFlags(0) {}
	__device__ constexpr SetOfTreeTypes(const SetOfTreeTypes &other) noexcept :
		treeTypeFlags(other.treeTypeFlags) {}

	// Returns if the set of tree types is empty.
	__device__ constexpr bool isEmpty() const noexcept {
		return !this->treeTypeFlags;
	}

	// Adds a tree type to the set.
	__device__ constexpr void add(const TreeType type) noexcept {
		this->treeTypeFlags |= static_cast<uint32_t>(1) << static_cast<uint32_t>(type);
	}

	// Returns if a specified tree type is contained within the set.
	__host__ __device__ constexpr bool contains(const TreeType type) const noexcept {
		return this->treeTypeFlags & (static_cast<uint32_t>(1) << static_cast<uint32_t>(type));
	}
};

// Returns if a specified biome under a specified version could theoretically contain a specified type of tree.
__device__ constexpr bool biomeContainsTreeType(const Biome biome, const TreeType treeType, const Version version) {
	if (treeType == TreeType::Unknown) return true;

	switch (biome) {
		case Biome::Forest:
			if (treeType == TreeType::Large_Oak) return version <= Version::v1_6_4 || Version::v1_8_9 < version;
			return treeType == TreeType::Oak || treeType == TreeType::Birch;
		case Biome::Birch_Forest:
			return treeType == TreeType::Birch;
		case Biome::Taiga:
			return treeType == TreeType::Pine || treeType == TreeType::Spruce;
		default:
			#if CUDA_IS_PRESENT
				return false;
			#else
				throw std::invalid_argument("Unsupported biome provided.");
			#endif
	}
}

// Given a Random instance, returns the next type of tree that would generate within the specified biome under the specified version.
// Random will be advanced 0-4 times.
__host__ __device__ TreeType getNextTreeType(Random &random, const Biome biome, const Version version) {
	switch (biome) {
		case Biome::Forest:
			if (version <= Version::v1_6_4) {
				if (!random.nextInt(5)) return TreeType::Birch;
				if (!random.nextInt(10)) return TreeType::Large_Oak;
			} else if (version <= Version::v1_8_9) {
				// Andrew: Not sure exactly in which versions this happens
				// NelS: Might also be 1.12.2.
				if (!random.nextInt(5)) return TreeType::Birch;
			} else if (version <= Version::v1_12_2) {
				if (!random.nextInt(5)) return TreeType::Birch;
				if (!random.nextInt(10)) return TreeType::Large_Oak;
			} else {
				if (random.nextFloat() < 0.2f) return TreeType::Birch;
				if (random.nextFloat() < 0.1f) return TreeType::Large_Oak;
			}
			return TreeType::Oak;
		case Biome::Birch_Forest:
			return TreeType::Birch;
		case Biome::Taiga:
			if (!random.nextInt(3)) return TreeType::Pine;
			return TreeType::Spruce;
		default:
			#if CUDA_IS_PRESENT
				return TreeType::Unknown;
			#else
				throw std::invalid_argument("Unsupported biome provided.");
			#endif
	}
}

/* Returns the minimum number of trees a specified biome under a specified version can ATTEMPT TO generate per population chunk.
   If the selected in-game world positions would be invalid for the trees, the number actually generated will be fewer.*/
__host__ __device__ constexpr int32_t biomeMinTreeCount(const Biome biome, const Version version) {
	switch (biome) {
		case Biome::Forest:
		case Biome::Birch_Forest:
		case Biome::Taiga:
			return 10;
		default:
			#if CUDA_IS_PRESENT
				return INT32_MIN;
			#else
				throw std::invalid_argument("Unsupported biome provided.");
			#endif
	}
}

// Returns the chance a specified biome under a specified version will generate an extra tree per population chunk.
__host__ __device__ constexpr float biomeExtraTreeChance(const Biome biome, const Version version) {
	return 0.1f;
}

/* Returns the maximum number of trees a specified biome under a specified version can ATTEMPT TO generate per population chunk.
   If the selected in-game world positions would be invalid for the trees, the number actually generated will be fewer.*/
__host__ __device__ constexpr int32_t biomeMaxTreeCount(const Biome biome, const Version version) {
	int32_t minTreeCount = biomeMinTreeCount(biome, version);
	#if CUDA_IS_PRESENT
		if (minTreeCount == INT32_MIN) return INT32_MIN;
	#endif
	// Andrew: Probably not 1.8.9
	if (version <= Version::v1_8_9) return minTreeCount + 1;
	return minTreeCount + static_cast<int32_t>(biomeExtraTreeChance(biome, version) != 0.0f);
}

/* Given a Random instance, returns the number of trees a specified biome under a specified version will ATTEMPT TO generate.
   If the selected in-game world positions would be invalid for the trees, the number actually generated will be fewer.
   Random will be advanced 1-2 times.*/
__host__ __device__ uint32_t biomeTreeCount(Random &random, const Biome biome, const Version version) {
	int32_t treeCount = biomeMinTreeCount(biome, version);

	if (version <= Version::v1_8_9) return treeCount + static_cast<int32_t>(!random.nextInt(10));
	return treeCount + static_cast<int32_t>(random.nextFloat() < biomeExtraTreeChance(biome, version));
}

// Returns the maximum number of PRNG calls a specified biome under a specified version can make.
__device__ constexpr int32_t biomeMaxRandomCalls(const Biome biome, const Version version) {
	switch (biome) {
		case Biome::Forest:
			return biomeMaxTreeCount(biome, version) * 21;
		case Biome::Birch_Forest:
			return biomeMaxTreeCount(biome, version) * 19;
		case Biome::Taiga:
			return biomeMaxTreeCount(biome, version) * 8;
		default:
			#if CUDA_IS_PRESENT
				return 0;
			#else
				throw std::invalid_argument("Unsupported biome provided.");
			#endif
	}
}

// Flattens the list of leaf states into a testable representation.
struct SetOfLeafStates {
	// Right-shifts 0-11 (<= 1.14.4) or 4-15 are 1 if the corresponding leaf was placed, or 0 otherwise.
	// Right-shifts 16-27 (<= 1.14.4) or 20-31 are 1 if the corresponding leaf is known, or 0 otherwise.
	uint32_t mask;

	__device__ constexpr SetOfLeafStates() noexcept : mask() {}
	__device__ constexpr SetOfLeafStates(const SetOfLeafStates &other) noexcept : mask(other.mask) {}
	__device__ constexpr SetOfLeafStates(const uint32_t mask) noexcept : mask(mask) {}
	__device__ constexpr SetOfLeafStates(const LeafState leafStates[NUMBER_OF_LEAF_POSITIONS], const Version version) : mask(0) {
		// Make sure all leaf states were given a valid value
		for (size_t i = 0; i < NUMBER_OF_LEAF_POSITIONS; i++) {
			if (leafStates[i] < LeafState::LeafWasNotPlaced || LeafState::Unknown < leafStates[i])
				#if CUDA_IS_PRESENT
					return;
				#else
					throw std::invalid_argument("Invalid leafstate in index" + std::to_string(i) + ".");
				#endif
		}
		for (int32_t y = 0; y < 3; y++) {
			for (int32_t xz = 0; xz < 4; xz++) {
				int32_t i = y * 4 + xz;
				// Andrew: Maybe 1.15, idk
				int32_t j = (version <= Version::v1_14_4 ? i : (3 - y) * 4 + xz);
				LeafState currentLeafState = leafStates[i];
				this->mask |= static_cast<uint32_t>(currentLeafState != LeafState::Unknown) << (16 + j); // Leaf is known
				this->mask |= static_cast<uint32_t>(currentLeafState == LeafState::LeafWasPlaced) << j;  // Leaf is present
			}
		}
	}

	// Given a Random instance and a specified version, returns whether the Random instance would generate the stored set of leaf states.
	// If true, Random will be advanced 16 times; otherwise Random will not be changed.
	__device__ bool canBeGeneratedBy(Random &random, const Version version) const noexcept {
		// Andrew: Only tested on 1.16.1
		if (Version::v1_14_4 < version && version <= Version::v1_16_1) random.skip<2>();
		if (((this->mask >> (16     ) & 1) && (this->mask       & 1) != Random(random).nextInt< 1>(2)) ||
			((this->mask >> (16 +  1) & 1) && (this->mask >>  1 & 1) != Random(random).nextInt< 2>(2)) ||
			((this->mask >> (16 +  2) & 1) && (this->mask >>  2 & 1) != Random(random).nextInt< 3>(2)) ||
			((this->mask >> (16 +  3) & 1) && (this->mask >>  3 & 1) != Random(random).nextInt< 4>(2)) ||
			((this->mask >> (16 +  4) & 1) && (this->mask >>  4 & 1) != Random(random).nextInt< 5>(2)) ||
			((this->mask >> (16 +  5) & 1) && (this->mask >>  5 & 1) != Random(random).nextInt< 6>(2)) ||
			((this->mask >> (16 +  6) & 1) && (this->mask >>  6 & 1) != Random(random).nextInt< 7>(2)) ||
			((this->mask >> (16 +  7) & 1) && (this->mask >>  7 & 1) != Random(random).nextInt< 8>(2)) ||
			((this->mask >> (16 +  8) & 1) && (this->mask >>  8 & 1) != Random(random).nextInt< 9>(2)) ||
			((this->mask >> (16 +  9) & 1) && (this->mask >>  9 & 1) != Random(random).nextInt<10>(2)) ||
			((this->mask >> (16 + 10) & 1) && (this->mask >> 10 & 1) != Random(random).nextInt<11>(2)) ||
			((this->mask >> (16 + 11) & 1) && (this->mask >> 11 & 1) != Random(random).nextInt<12>(2)) ||
			((this->mask >> (16 + 12) & 1) && (this->mask >> 12 & 1) != Random(random).nextInt<13>(2)) ||
			((this->mask >> (16 + 13) & 1) && (this->mask >> 13 & 1) != Random(random).nextInt<14>(2)) ||
			((this->mask >> (16 + 14) & 1) && (this->mask >> 14 & 1) != Random(random).nextInt<15>(2)) ||
			((this->mask >> (16 + 15) & 1) && (this->mask >> 15 & 1) != Random(random).nextInt<16>(2))) return false;
		random.skip<16>();
		// #pragma unroll
		// for (int32_t i = 0; i < 16; ++i) {
		//     if ((this->mask >> (16 + i) & 1) && (this->mask >> i & 1) != random.nextInt<1>(2)) {
		//         #pragma unroll
		//         for (int32_t j = 0; j < i; ++j) random.skip<-1>();
		//         return false;
		//     }
		// }

		return true;
	}

	// Returns the number of bits of information that can be derived from the known leaves.
	__host__ __device__ constexpr double getEstimatedBits() const noexcept {
		return static_cast<double>(getNumberOfOnesIn(this->mask >> 16));
	}
};

struct TrunkHeight {
	// Returns the next pseudorandom integer in the range [a, a + b], uniformly distributed.
	// Random will be advanced 1-2 times.
	__host__ __device__ static uint32_t getNextValueInRange(Random &random, const uint32_t a, const uint32_t b) noexcept {
		return a + random.nextInt(b + 1);
	}

	// Returns the next pseudorandom integer in the range [a, a + b + c].
	// If c == 0, this is equivalent to ::getNextValueInRange(random, a, b) with an extra state advancement. Random will be advanced 2-3 times.
	// Otherwise the result is triangularly(?) distributed with (b+c)/2 as the highest point. Random will be advanced 2-4 times.
	__host__ __device__ static uint32_t getNextValueInRange(Random &random, const uint32_t a, const uint32_t b, const uint32_t c) noexcept {
		return a + random.nextInt(b + 1) + random.nextInt(c + 1);
	}
};

struct OakAttributes {
	PossibleHeightsRange trunkHeight;
	static constexpr PossibleHeightsRange TRUNK_HEIGHT_BOUNDS = {4, 6};
	SetOfLeafStates leafStates;

	__device__ constexpr OakAttributes() noexcept :
		trunkHeight(OakAttributes::TRUNK_HEIGHT_BOUNDS),
		leafStates() {}
	__device__ constexpr OakAttributes(const OakAttributes &other) noexcept :
		trunkHeight(other.trunkHeight, OakAttributes::TRUNK_HEIGHT_BOUNDS),
		leafStates(other.leafStates) {}
	__device__ constexpr OakAttributes(const PossibleHeightsRange &trunkHeight, const SetOfLeafStates &leafStates) noexcept :
		trunkHeight(trunkHeight, OakAttributes::TRUNK_HEIGHT_BOUNDS),
		leafStates(leafStates) {}

	// Returns whether the given random instance under the specified version would generate an oak tree with attributes matching the initialized one.
	// The Random instance will be advanced 1-2 times (< 1.14.4) or 2-3 times otherwise.
	__device__ bool canBeGeneratedBy(Random &random, const Version version) const noexcept {
		// Andrew: Maybe 1.15, idk
		if (version <= Version::v1_14_4) {
			if (!this->trunkHeight.contains(TrunkHeight::getNextValueInRange(random, 4, 2))) return false;
		} else {
			if (!this->trunkHeight.contains(TrunkHeight::getNextValueInRange(random, 4, 2, 0))) return false;
		}
		return this->leafStates.canBeGeneratedBy(random, version);
	}

	// Returns the number of bits of information that can be derived from the attributes.
	__host__ __device__ constexpr double getEstimatedBits() const noexcept {
		return constexprLog2(static_cast<double>(OakAttributes::TRUNK_HEIGHT_BOUNDS.getRange())/static_cast<double>(trunkHeight.getRange())) +
			this->leafStates.getEstimatedBits();
	}

	/* Fast-forwards the Random instance, as if a tree under the specified version had been generated, without actually
	   calculating its attributes.
	   A tree's generation can abort early (causing fewer Random advancements) if the in-game world is unsuitable--dirt/grass
	   is not present, another tree is too close, etc. If this is the case, set isValidIngamePosition to false.*/
	__device__ static void skip(Random &random, const bool isValidIngamePosition, const Version version) noexcept {
		// Andrew: Maybe 1.15, idk
		if (version <= Version::v1_14_4) TrunkHeight::getNextValueInRange(random, 4, 2);
		else {
			TrunkHeight::getNextValueInRange(random, 4, 2, 0);
			if (version <= Version::v1_16_1) {
				// Andrew: 1st extra leaf call
				random.skip<1>();
			}
		}

		if (isValidIngamePosition) {
			if (version <= Version::v1_14_4) random.skip<16>();
			else if (version <= Version::v1_16_1) {
				// Andrew: + 2nd extra leaf call + beehive
				random.skip<18>();
			} else {
				// Andrew: + beehive
				random.skip<17>();
			}
		}
	}
};

struct LargeOakAttributes {
	PossibleHeightsRange trunkHeight;
	static constexpr PossibleHeightsRange TRUNK_HEIGHT_BOUNDS = {3, 14};

	__device__ constexpr LargeOakAttributes() noexcept :
		trunkHeight(LargeOakAttributes::TRUNK_HEIGHT_BOUNDS) {}
	__device__ constexpr LargeOakAttributes(const LargeOakAttributes &other) noexcept :
		trunkHeight(other.trunkHeight, LargeOakAttributes::TRUNK_HEIGHT_BOUNDS) {}
	__device__ constexpr LargeOakAttributes(const PossibleHeightsRange &trunkHeight) noexcept :
		trunkHeight(trunkHeight, LargeOakAttributes::TRUNK_HEIGHT_BOUNDS) {}

	// Returns whether the given random instance under the specified version would generate an oak tree with attributes matching the initialized one.
	// The Random instance will be advanced 2-3 times.
	__device__ bool canBeGeneratedBy(Random &random, const Version version) const noexcept {
		// Andrew: Maybe 1.15, idk
		// if (version <= Version::v1_14_4) {

		// } else {
		//     if (!this->trunkHeight.contains(TrunkHeight::getNextValueInRange(random, 3, 11, 0))) return false;
		// }
		// return true;
		return version <= Version::v1_14_4 || this->trunkHeight.contains(TrunkHeight::getNextValueInRange(random, 3, 11, 0));
	}

	__device__ static float treeShape(const int32_t n, const int32_t n2) noexcept {
		if ((float)n2 < (float)n * 0.3f) return -1.0f;
		float f = (float)n / 2.0f;
		float f2 = f - (float)n2;
		float f3 = sqrt(f * f - f2 * f2);
		if (f2 == 0.0f) f3 = f;
		else if (abs(f2) >= f) return 0.0f;
		return f3 * 0.5f;
	}

	// Returns the number of bits of information that can be derived from the attributes.
	__host__ __device__ constexpr double getEstimatedBits() const noexcept {
		return constexprLog2(static_cast<double>(LargeOakAttributes::TRUNK_HEIGHT_BOUNDS.getRange())/static_cast<double>(trunkHeight.getRange()));
	}

	/* Fast-forwards the Random instance, as if a tree under the specified version had been generated, without actually
	   calculating its attributes.
	   A tree's generation can abort early (causing fewer Random advancements) if the in-game world is unsuitable--dirt/grass
	   is not present, another tree is too close, etc. If this is the case, set isValidIngamePosition to false.*/
	__device__ static void skip(Random &random, const bool isValidIngamePosition, const Version version) {
		// Andrew: Maybe 1.15, idk
		if (version <= Version::v1_14_4) random.skip<2>();
		else if (version <= Version::v1_16_1) {
			uint32_t trunkHeight = TrunkHeight::getNextValueInRange(random, 3, 11, 0);

			// Andrew: extra leaf calls
			random.skip<1>();

			if (isValidIngamePosition) {
				// NelS: Woah, oh, oh, it's MAGIC!
				uint32_t branchCount = (0x765543321000 >> ((trunkHeight - 3) * 4)) & 0xF;
				uint32_t blobCount = 1;
				for (uint32_t i = 0; i < branchCount; i++) {
					int32_t n4 = trunkHeight + 2;
					int32_t n2 = n4 - 5 - i;
					float f = treeShape(n4, n2);
					double d4 = 1.0 * (double)f * ((double)random.nextFloat() + 0.328);
					double d2 = (double)(random.nextFloat() * 2.0f) * 3.141592653589793;
					int32_t ex = floor(d4 * sin(d2) + 0.5);
					int32_t ez = floor(d4 * cos(d2) + 0.5);
					int32_t ey = n2 - 1;
					int32_t sy = ey - sqrt((double)(ex * ex + ez * ez)) * 0.381;
					if (sy >= n4 * 0.2) ++blobCount;
				}
				if (blobCount & 8) random.skip<8>();
				if (blobCount & 4) random.skip<4>();
				if (blobCount & 2) random.skip<2>();
				if (blobCount & 1) random.skip<1>();
				// Andrew: Beehive 1
				random.skip<1>();
			}
		} else {
			uint32_t trunkHeight = TrunkHeight::getNextValueInRange(random, 3, 11, 0);

			if (isValidIngamePosition) {
				// Andrew: Trunk 0 - 14
				uint32_t branchCount = (0x765543321000 >> (4*trunkHeight - 12)) & 0xf;
				if (branchCount & 4) random.skip<8>();
				if (branchCount & 2) random.skip<4>();
				if (branchCount & 1) random.skip<2>();
				// Andrew: Beehive 1
				random.skip<1>();
			}
		}
	}
};

struct BirchAttributes {
	PossibleHeightsRange trunkHeight;
	static constexpr PossibleHeightsRange TRUNK_HEIGHT_BOUNDS = {5, 7};
	SetOfLeafStates leafStates;

	__device__ constexpr BirchAttributes() :
		trunkHeight(BirchAttributes::TRUNK_HEIGHT_BOUNDS),
		leafStates() {}
	__device__ constexpr BirchAttributes(const BirchAttributes &other) :
		trunkHeight(other.trunkHeight, BirchAttributes::TRUNK_HEIGHT_BOUNDS),
		leafStates(other.leafStates) {}
	__device__ constexpr BirchAttributes(const PossibleHeightsRange &trunkHeight, const SetOfLeafStates &leafStates) :
		trunkHeight(trunkHeight, BirchAttributes::TRUNK_HEIGHT_BOUNDS),
		leafStates(leafStates) {}

	__device__ bool canBeGeneratedBy(Random &random, const Version version) const {
		// Andrew: Maybe 1.15, idk
		if (version <= Version::v1_14_4) {
			if (!this->trunkHeight.contains(TrunkHeight::getNextValueInRange(random, 5, 2))) return false;
		} else {
			if (!this->trunkHeight.contains(TrunkHeight::getNextValueInRange(random, 5, 2, 0))) return false;
		}
		return this->leafStates.canBeGeneratedBy(random, version);
	}

	// Returns the number of bits of information that can be derived from the attributes.
	__host__ __device__ constexpr double getEstimatedBits() const noexcept {
		return constexprLog2(static_cast<double>(BirchAttributes::TRUNK_HEIGHT_BOUNDS.getRange())/static_cast<double>(trunkHeight.getRange())) +
			this->leafStates.getEstimatedBits();
	}

	/* Fast-forwards the Random instance, as if a tree under the specified version had been generated, without actually
	   calculating its attributes.
	   A tree's generation can abort early (causing fewer Random advancements) if the in-game world is unsuitable--dirt/grass
	   is not present, another tree is too close, etc. If this is the case, set isValidIngamePosition to false.*/
	__device__ static void skip(Random &random, const bool isValidIngamePosition, const Version version) {
		// Andrew: Maybe 1.15, idk
		if (version <= Version::v1_14_4) {
			TrunkHeight::getNextValueInRange(random, 5, 2);
		} else {
			TrunkHeight::getNextValueInRange(random, 5, 2, 0);
		}

		if (Version::v1_14_4 < version && version <= Version::v1_16_1) {
			// Andrew: 1st extra leaf call
			random.skip<1>();
		}

		if (isValidIngamePosition) {
			if (version <= Version::v1_14_4) {
				random.skip<16>();
			} else if (version <= Version::v1_16_1) {
				// Andrew: + 2nd extra leaf call + beehive
				random.skip<18>();
			} else {
				// Andrew: + beehive
				random.skip<17>();
			}
		}
	}
};

// height: `total_height - 1` == `trunk_height + 1`
// leavesHeight: `leavesHeight - 1` except for if the last 2 leaf radii are the same (in which case the last layer generates but has radius 0).
// leavesWidestRadius: `leavesWidestRadius` of the widest layer
struct PineAttributes {
	PossibleHeightsRange trunkHeight;
	static constexpr PossibleHeightsRange TRUNK_HEIGHT_BOUNDS = {7, 11}; // Adjusted from visible [6,10]
	PossibleHeightsRange leavesHeight;
	static constexpr PossibleHeightsRange LEAVES_HEIGHT_BOUNDS = {3, 4}; // Adjusted from (mostly) visible [4,5]
	PossibleRadiiRange leavesWidestRadius;
	static constexpr PossibleRadiiRange LEAVES_WIDEST_RADIUS_BOUNDS = {1, 3};

	__device__ constexpr PineAttributes() :
		trunkHeight(PineAttributes::TRUNK_HEIGHT_BOUNDS),
		leavesHeight(PineAttributes::LEAVES_HEIGHT_BOUNDS),
		leavesWidestRadius(PineAttributes::LEAVES_WIDEST_RADIUS_BOUNDS) {}
	__device__ constexpr PineAttributes(const PineAttributes &other) :
		trunkHeight(other.trunkHeight, PineAttributes::TRUNK_HEIGHT_BOUNDS),
		leavesHeight(other.leavesHeight, PineAttributes::LEAVES_HEIGHT_BOUNDS),
		leavesWidestRadius(other.leavesWidestRadius, PineAttributes::LEAVES_WIDEST_RADIUS_BOUNDS) {}
	__device__ constexpr PineAttributes(const PossibleHeightsRange &trunkHeight, const PossibleHeightsRange &leavesHeight, const PossibleRadiiRange &leavesWidestRadius) :
		trunkHeight({trunkHeight.lowerBound + 1, trunkHeight.upperBound + 1}, PineAttributes::TRUNK_HEIGHT_BOUNDS),
		leavesHeight({leavesHeight.lowerBound - 1, leavesHeight.upperBound - 1}, PineAttributes::LEAVES_HEIGHT_BOUNDS),
		leavesWidestRadius(leavesWidestRadius, PineAttributes::LEAVES_WIDEST_RADIUS_BOUNDS) {}

	__device__ bool canBeGeneratedBy(Random &random, const Version version) const {
		if (!this->trunkHeight.contains(TrunkHeight::getNextValueInRange(random, 7, 4))) return false;
		uint32_t generatedLeavesHeight = 3 + random.nextInt(2);
		return this->leavesHeight.contains(generatedLeavesHeight) && this->leavesWidestRadius.contains(constexprMin(generatedLeavesHeight - 1, 1 + random.nextInt(generatedLeavesHeight + 1)));
	}

	// Returns the number of bits of information that can be derived from the attributes.
	__host__ __device__ constexpr double getEstimatedBits() const noexcept {
		return constexprLog2(static_cast<double>(PineAttributes::TRUNK_HEIGHT_BOUNDS.getRange())/static_cast<double>(trunkHeight.getRange())) +
			constexprLog2(static_cast<double>(PineAttributes::LEAVES_HEIGHT_BOUNDS.getRange())/static_cast<double>(leavesHeight.getRange())) +
			constexprLog2(static_cast<double>(PineAttributes::LEAVES_WIDEST_RADIUS_BOUNDS.getRange())/static_cast<double>(leavesWidestRadius.getRange()));
	}

	/* Fast-forwards the Random instance, as if a tree under the specified version had been generated, without actually
	   calculating its attributes.
	   A tree's generation can abort early (causing fewer Random advancements) if the in-game world is unsuitable--dirt/grass
	   is not present, another tree is too close, etc. If this is the case, set isValidIngamePosition to false.*/
	__device__ static void skip(Random &random, const bool isValidIngamePosition, const Version version) {
		TrunkHeight::getNextValueInRange(random, 7, 4);
		uint32_t generatedLeavesHeight = 3 + random.nextInt(2);
		// uint32_t generatedLeavesWidestRadius = 1 + random.nextInt(generatedLeavesHeight + 1);
		random.nextInt(generatedLeavesHeight + 1);
	}
};

// height: `total_height - 1`
// logsBelowBottommostLeaves: `logsBelowBottommostLeaves` number of logs without leaves starting from ground
// leavesWidestRadius: `leavesWidestRadius` of the widest layer
// top_leaves_radius: `top_leaves_radius`
// trunk_reduction: `trunk_reduction`
struct SpruceAttributes {
	PossibleHeightsRange trunkHeight;
	static constexpr PossibleHeightsRange TRUNK_HEIGHT_BOUNDS = {6, 9}; // Adjusted from visible [4,9]
	PossibleHeightsRange logsBelowBottommostLeaves;
	static constexpr PossibleHeightsRange LOGS_BELOW_BOTTOMMOST_LEAVES_BOUNDS = {1, 2};
	PossibleHeightsRange leavesAboveTrunk;
	static constexpr PossibleHeightsRange LEAVES_ABOVE_TRUNK_BOUNDS = {0, 2}; // Adjusted from visible [1,3]
	PossibleRadiiRange leavesWidestRadius;
	static constexpr PossibleRadiiRange LEAVES_WIDEST_RADIUS_BOUNDS = {2, 3};
	PossibleRadiiRange topmostLeavesRadius;
	static constexpr PossibleRadiiRange TOPMOST_LEAVES_RADIUS_BOUNDS = {0, 1};

	__device__ constexpr SpruceAttributes() :
		trunkHeight(SpruceAttributes::TRUNK_HEIGHT_BOUNDS),
		logsBelowBottommostLeaves(SpruceAttributes::LOGS_BELOW_BOTTOMMOST_LEAVES_BOUNDS),
		leavesAboveTrunk(SpruceAttributes::LEAVES_ABOVE_TRUNK_BOUNDS),
		leavesWidestRadius(SpruceAttributes::LEAVES_WIDEST_RADIUS_BOUNDS),
		topmostLeavesRadius(SpruceAttributes::TOPMOST_LEAVES_RADIUS_BOUNDS) {}
	__device__ constexpr SpruceAttributes(const SpruceAttributes &other) :
		trunkHeight(other.trunkHeight, SpruceAttributes::TRUNK_HEIGHT_BOUNDS),
		logsBelowBottommostLeaves(other.logsBelowBottommostLeaves, SpruceAttributes::LOGS_BELOW_BOTTOMMOST_LEAVES_BOUNDS),
		leavesAboveTrunk(other.leavesAboveTrunk, SpruceAttributes::LEAVES_ABOVE_TRUNK_BOUNDS),
		leavesWidestRadius(other.leavesWidestRadius, SpruceAttributes::LEAVES_WIDEST_RADIUS_BOUNDS),
		topmostLeavesRadius(other.topmostLeavesRadius, SpruceAttributes::TOPMOST_LEAVES_RADIUS_BOUNDS) {}
	__device__ constexpr SpruceAttributes(const PossibleHeightsRange &trunkHeight, const PossibleHeightsRange &logsBelowBottommostLeaves, const PossibleRadiiRange &leavesWidestRadius, const PossibleRadiiRange &topmostLeavesRadius, const PossibleRadiiRange &leavesAboveTrunk) :
		trunkHeight({trunkHeight.lowerBound + leavesAboveTrunk.lowerBound - 1, trunkHeight.upperBound + leavesAboveTrunk.upperBound - 1}, SpruceAttributes::TRUNK_HEIGHT_BOUNDS),
		logsBelowBottommostLeaves(logsBelowBottommostLeaves, SpruceAttributes::LOGS_BELOW_BOTTOMMOST_LEAVES_BOUNDS),
		leavesAboveTrunk({leavesAboveTrunk.lowerBound - 1, leavesAboveTrunk.upperBound - 1}, SpruceAttributes::LEAVES_ABOVE_TRUNK_BOUNDS),
		leavesWidestRadius(leavesWidestRadius, SpruceAttributes::LEAVES_WIDEST_RADIUS_BOUNDS),
		topmostLeavesRadius(topmostLeavesRadius, SpruceAttributes::TOPMOST_LEAVES_RADIUS_BOUNDS) {}

	__device__ bool canBeGeneratedBy(Random &random, const Version version) const {
		uint32_t trunkHeight = TrunkHeight::getNextValueInRange(random, 6, 3);
		if (!this->trunkHeight.contains(trunkHeight)) return false;
		uint32_t logsBelowBottommostLeaves = 1 + random.nextInt(2);
		if (!this->logsBelowBottommostLeaves.contains(logsBelowBottommostLeaves)) return false;
		uint32_t maximumPossibleLeavesRadius = 2 + random.nextInt(2);
		uint32_t topmostLeavesRadius = random.nextInt(2);
		return this->topmostLeavesRadius.contains(topmostLeavesRadius) && this->leavesWidestRadius.contains(constexprMin(2 + static_cast<uint32_t>(trunkHeight + 1 - logsBelowBottommostLeaves + topmostLeavesRadius >= 8), maximumPossibleLeavesRadius)) && this->leavesAboveTrunk.contains(random.nextInt(3));
	}

	// Returns the number of bits of information that can be derived from the attributes.
	__host__ __device__ constexpr double getEstimatedBits() const noexcept {
		return constexprLog2(static_cast<double>(SpruceAttributes::TRUNK_HEIGHT_BOUNDS.getRange())/static_cast<double>(trunkHeight.getRange())) +
			constexprLog2(static_cast<double>(SpruceAttributes::LOGS_BELOW_BOTTOMMOST_LEAVES_BOUNDS.getRange())/static_cast<double>(logsBelowBottommostLeaves.getRange())) +
			constexprLog2(static_cast<double>(SpruceAttributes::LEAVES_ABOVE_TRUNK_BOUNDS.getRange())/static_cast<double>(leavesAboveTrunk.getRange())) +
			constexprLog2(static_cast<double>(SpruceAttributes::LEAVES_WIDEST_RADIUS_BOUNDS.getRange())/static_cast<double>(leavesWidestRadius.getRange())) +
			constexprLog2(static_cast<double>(SpruceAttributes::TOPMOST_LEAVES_RADIUS_BOUNDS.getRange())/static_cast<double>(topmostLeavesRadius.getRange()));
	}

	__device__ static void skip(Random &random, const bool isValidIngamePosition, const Version version) {
		if (isValidIngamePosition) {
			random.skip<4>();
			random.nextInt(3);
		} else random.skip<3>();
	}
};

struct TreeChunkPosition {
	uint32_t populationChunkXOffset, populationChunkZOffset;
	static constexpr IntInclusiveRange POPULATION_CHUNK_OFFSET_BOUNDS = {0, 15};
	SetOfTreeTypes possibleTreeTypes;
	// union {
		OakAttributes oakAttributes;
		LargeOakAttributes largeOakAttributes;
		BirchAttributes birchAttributes;
		PineAttributes pineAttributes;
		SpruceAttributes spruceAttributes;
	// };

	__device__ constexpr TreeChunkPosition() noexcept :
		populationChunkXOffset(this->POPULATION_CHUNK_OFFSET_BOUNDS.lowerBound),
		populationChunkZOffset(this->POPULATION_CHUNK_OFFSET_BOUNDS.lowerBound),
		possibleTreeTypes() {}
	__device__ constexpr TreeChunkPosition(const TreeChunkPosition &other) noexcept :
		populationChunkXOffset(constexprMin(constexprMax(other.populationChunkXOffset, static_cast<uint32_t>(this->POPULATION_CHUNK_OFFSET_BOUNDS.lowerBound)), static_cast<uint32_t>(this->POPULATION_CHUNK_OFFSET_BOUNDS.upperBound))),
		populationChunkZOffset(constexprMin(constexprMax(other.populationChunkZOffset, static_cast<uint32_t>(this->POPULATION_CHUNK_OFFSET_BOUNDS.lowerBound)), static_cast<uint32_t>(this->POPULATION_CHUNK_OFFSET_BOUNDS.upperBound))),
		possibleTreeTypes(other.possibleTreeTypes),
		oakAttributes(other.oakAttributes),
		largeOakAttributes(other.largeOakAttributes),
		birchAttributes(other.birchAttributes),
		pineAttributes(other.pineAttributes),
		spruceAttributes(other.spruceAttributes) {}

	// Returns if a Random instance, given a specified biome and version, would generate a tree matching the specified tree's x-coordinate, z-coordinate, and possible types.
	// Random will be advanced 1-6 times.
	__device__ bool testXZAndType(Random &random, const Biome biome, const Version version) const {
		return random.nextInt(16) == this->populationChunkXOffset && random.nextInt(16) == this->populationChunkZOffset && (this->possibleTreeTypes.isEmpty() || this->possibleTreeTypes.contains(getNextTreeType(random, biome, version)));
	}

	// Returns if a Random instance, given a specified biome and version, would generate a tree matching the specified tree's possible types and type-specific attributes.
	__device__ bool testTypeAndAttributes(Random &random, const Biome biome, const Version version) const {
		if (this->possibleTreeTypes.isEmpty()) return true;
		TreeType type = getNextTreeType(random, biome, version);
		if (!this->possibleTreeTypes.contains(type)) return false;

		switch (type) {
			case TreeType::Oak:       return this->oakAttributes.canBeGeneratedBy(random, version);
			case TreeType::Large_Oak: return this->largeOakAttributes.canBeGeneratedBy(random, version);
			case TreeType::Birch:     return this->birchAttributes.canBeGeneratedBy(random, version);
			case TreeType::Pine:      return this->pineAttributes.canBeGeneratedBy(random, version);
			case TreeType::Spruce:    return this->spruceAttributes.canBeGeneratedBy(random, version);
			default:
				#if CUDA_IS_PRESENT
					return false;
				#else
					throw std::invalid_argument("Unsupported tree type provided.");
				#endif
		}
	}

	// Returns if a Random instance, given a specified biome and version, would generate a tree matching the specified tree's z-coordinate, possible types, and type-specific attributes.
	__device__ bool testZTypeAndAttributes(Random &random, const Biome biome, const Version version) const {
		if (random.nextInt(16) != this->populationChunkZOffset) return false;
		return this->testTypeAndAttributes(random, biome, version);
	}

	// Returns if a Random instance, given a specified biome and version, would generate a tree matching the specified tree's x-coordinate, z-coordinate, type, and type-specific attributes.
	__device__ bool testXZTypeAndAttributes(Random &random, const Biome biome, const Version version) const {
		return random.nextInt(16) == this->populationChunkXOffset && this->testZTypeAndAttributes(random, biome, version);
	}

	// Given a specified biome and version, returns the number of bits of information that can be derived from the known tree data.
	// If multiple tree types are contained in possibleTreeTypes, the combined number of bits will be averaged over the feasible tree types.
	__host__ __device__ constexpr double getEstimatedBits(const Biome biome, const Version version) const noexcept {
		double attributeBits = 0.;
		uint32_t treeTypeCounter = 0;
		switch (biome) {
			case Biome::Forest:
				if (this->possibleTreeTypes.contains(TreeType::Oak)) {
					++treeTypeCounter;
					attributeBits += this->oakAttributes.getEstimatedBits() + constexprLog2(5./4.); // nextInt(5) != 0
					if (version <= Version::v1_6_4 || Version::v1_8_9 < version) attributeBits += constexprLog2(10./9.); // nextInt(10) != 0
				}
				if (this->possibleTreeTypes.contains(TreeType::Large_Oak)) {
					++treeTypeCounter;
					attributeBits += this->largeOakAttributes.getEstimatedBits() + constexprLog2(5./4.) + constexprLog2(10.); // nextInt(5) != 0, then nextInt(10) == 0
				}
				if (this->possibleTreeTypes.contains(TreeType::Birch)) {
					++treeTypeCounter;
					attributeBits += this->birchAttributes.getEstimatedBits() + constexprLog2(5.); // nextInt(5) == 0
				}
				break;
			case Biome::Birch_Forest:
				if (this->possibleTreeTypes.contains(TreeType::Birch)) {
					++treeTypeCounter;
					attributeBits += this->birchAttributes.getEstimatedBits();
				}
				break;
			case Biome::Taiga:
				if (this->possibleTreeTypes.contains(TreeType::Pine)) {
					++treeTypeCounter;
					attributeBits += this->pineAttributes.getEstimatedBits() + constexprLog2(3.); // nextInt(3) == 0
				}
				break;
				if (this->possibleTreeTypes.contains(TreeType::Spruce)) {
					++treeTypeCounter;
					attributeBits += this->spruceAttributes.getEstimatedBits() + constexprLog2(3./2.); // nextInt(3) != 0
				}
				break;
		}
		return treeTypeCounter ? attributeBits/static_cast<double>(treeTypeCounter) + 8. : 0.; // nextInt(16), nextInt(16)

	}

	/* Fast-forwards the Random instance, as if a tree under the specified biome and version had been generated, without actually 
	   calculating its attributes.
	   A tree's generation can abort early (causing fewer Random advancements) if the in-game world is unsuitable--dirt/grass
	   is not present, another tree is too close, etc. If this is the case, set isValidIngamePosition to false.*/
	__device__ static void skip(Random &random, const Biome biome, const bool isValidIngamePosition, const Version version) {
		// uint32_t populationChunkXOffset = random.nextInt(16);
		// uint32_t populationChunkZOffset = random.nextInt(16);
		random.skip<2>();

		switch (getNextTreeType(random, biome, version)) {
			case TreeType::Oak:
				OakAttributes::skip(random, isValidIngamePosition, version);
				break;
			case TreeType::Large_Oak:
				LargeOakAttributes::skip(random, isValidIngamePosition, version);
				break;
			case TreeType::Birch:
				BirchAttributes::skip(random, isValidIngamePosition, version);
				break;
			case TreeType::Pine:
				PineAttributes::skip(random, isValidIngamePosition, version);
				break;
			case TreeType::Spruce:
				SpruceAttributes::skip(random, isValidIngamePosition, version);
				break;
			default:
				#if CUDA_IS_PRESENT
					return;
				#else
					throw std::invalid_argument("Unsupported tree type provided.");
				#endif
		}
	}
};

struct TreeChunk {
	Version version;
	Biome biome;
	int32_t populationChunkX, populationChunkZ;
	TreeChunkPosition treePositions[16];
	uint32_t numberOfTreePositions;
	int32_t maxCalls, maxTreeCount, rangeOfPossibleSkips;
	uint64_t salt;
	// Might only need this for first population chunk instead of all population chunks, in which case it'd be better to have this as a global constexpr variable
	// bool collapseNearbySeedsFlag;

	__device__ constexpr TreeChunk() noexcept :
		version(),
		biome(),
		populationChunkX(),
		populationChunkZ(),
		treePositions(),
		numberOfTreePositions(0),
		maxCalls(),
		maxTreeCount(),
		rangeOfPossibleSkips(),
		salt()
		// , collapseNearbySeedsFlag(),
		{}
	__device__ constexpr TreeChunk(const Biome biome, const Version version) noexcept :
		version(version),
		biome(biome),
		populationChunkX(),
		populationChunkZ(),
		treePositions(),
		numberOfTreePositions(0),
		maxCalls(biomeMaxRandomCalls(biome, version)),
		maxTreeCount(biomeMaxTreeCount(biome, version)),
		rangeOfPossibleSkips(1 + (version <= Version::v1_12_2 ? 10000 : maxCalls)),
		// Andrew:
		// 1.14.4 Forest - 60001
		// 1.16.4 Forest - 80001 ??? Not sure if that's what it was
		salt(version <= Version::v1_14_4 ? 60001 : 80001)
		// , collapseNearbySeedsFlag(version <= Version::v1_12_2),
		{}
	__device__ constexpr TreeChunk(const Biome biome, const Version version, const int32_t populationChunkX, const int32_t populationChunkZ) noexcept :
		version(version),
		biome(biome),
		populationChunkX(populationChunkX),
		populationChunkZ(populationChunkZ),
		treePositions(),
		numberOfTreePositions(0),
		maxCalls(biomeMaxRandomCalls(biome, version)),
		maxTreeCount(biomeMaxTreeCount(biome, version)),
		rangeOfPossibleSkips(1 + (version <= Version::v1_12_2 ? 10000 : maxCalls)),
		// Andrew:
		// 1.14.4 Forest - 60001
		// 1.16.4 Forest - 80001 ??? Not sure if that's what it was
		salt(version <= Version::v1_14_4 ? 60001 : 80001)
		// , collapseNearbySeedsFlag(version <= Version::v1_12_2)
		{}

	__device__ constexpr TreeChunkPosition &createNewTree(const int32_t x, const int32_t z, const TreeType treeType) {
		if (!biomeContainsTreeType(this->biome, treeType, this->version)) {
			// Invalid tree type for the given biome
			#if CUDA_IS_PRESENT
				return this->treePositions[sizeof(this->treePositions)/sizeof(*this->treePositions) - 1];
			#else
				throw std::invalid_argument("The specified tree type cannot generate under the TreeChunk's set biome and version.");
			#endif
		}

		uint32_t populationChunkX = getPopulationChunkCoordinate(x, version);
		uint32_t populationChunkZ = getPopulationChunkCoordinate(z, version);
		if (!this->numberOfTreePositions) {
			this->populationChunkX = populationChunkX;
			this->populationChunkZ = populationChunkZ;
		} else if (populationChunkX != this->populationChunkX || populationChunkZ != this->populationChunkZ) {
			#if CUDA_IS_PRESENT
				return this->treePositions[sizeof(this->treePositions)/sizeof(*this->treePositions) - 1];
			#else
				throw std::invalid_argument("New tree does not fall within the specified tree chunk.");
			#endif
		}
		uint32_t populationChunkXOffset = getOffsetWithinPopulationChunk(x, version);
		uint32_t populationChunkZOffset = getOffsetWithinPopulationChunk(z, version);

		for (uint32_t i = 0; i < this->numberOfTreePositions; i++) {
			TreeChunkPosition &tree = this->treePositions[i];
			if (tree.populationChunkXOffset == populationChunkXOffset && tree.populationChunkZOffset == populationChunkZOffset) {
				if (treeType == TreeType::Unknown) {
					#if CUDA_IS_PRESENT
						return this->treePositions[sizeof(this->treePositions)/sizeof(*this->treePositions) - 1];
					#else
						throw std::invalid_argument("A tree with known type cannot be superseded with a tree of unknown type.");
					#endif
				}
				if (tree.possibleTreeTypes.contains(treeType)) {
					#if CUDA_IS_PRESENT
						return this->treePositions[sizeof(this->treePositions)/sizeof(*this->treePositions) - 1];
					#else
						throw std::invalid_argument("Encountered duplicate tree with duplicate coordinates, biome, version, and type.");
					#endif
				}
				tree.possibleTreeTypes.add(treeType);
				return tree;
			}
		}

		TreeChunkPosition &tree = this->treePositions[this->numberOfTreePositions];
		++this->numberOfTreePositions;
		tree.populationChunkXOffset = populationChunkXOffset;
		tree.populationChunkZOffset = populationChunkZOffset;
		if (treeType != TreeType::Unknown) tree.possibleTreeTypes.add(treeType);
		return tree;
	}

	__device__ constexpr void addTree(const InputData &treeData) {
		switch (treeData.treeType) {
			case TreeType::Oak:
				this->createNewTree(treeData.coordinate.x, treeData.coordinate.z, TreeType::Oak).oakAttributes = OakAttributes(treeData.trunkHeight, SetOfLeafStates(treeData.leafStates, this->version));
				break;
			case TreeType::Large_Oak:
				this->createNewTree(treeData.coordinate.x, treeData.coordinate.z, TreeType::Large_Oak).largeOakAttributes = LargeOakAttributes(treeData.trunkHeight);
				break;
			case TreeType::Birch:
				this->createNewTree(treeData.coordinate.x, treeData.coordinate.z, TreeType::Birch).birchAttributes = BirchAttributes(treeData.trunkHeight, SetOfLeafStates(treeData.leafStates, this->version));
				break;
			case TreeType::Pine:
				this->createNewTree(treeData.coordinate.x, treeData.coordinate.z, TreeType::Pine).pineAttributes = PineAttributes(treeData.trunkHeight, treeData.leavesHeight, treeData.pineLeavesWidestRadius);
				break;
			case TreeType::Spruce:
				this->createNewTree(treeData.coordinate.x, treeData.coordinate.z, TreeType::Spruce).spruceAttributes = SpruceAttributes(treeData.trunkHeight, treeData.logsBelowBottommostLeaves, treeData.spruceLeavesWidestRadius, treeData.topmostLeavesRadius, treeData.leavesAboveTrunk);
				break;
			case TreeType::Unknown:
				this->createNewTree(treeData.coordinate.x, treeData.coordinate.z, TreeType::Unknown);
				break;
			default:
				#if CUDA_IS_PRESENT
					return;
				#else
					throw std::invalid_argument("Unsupported tree type provided.");
				#endif
		}
	}

	__host__ __device__ constexpr double getEstimatedBits() const noexcept {
		double bits = 0.;
		for (uint32_t i = 0; i < this->numberOfTreePositions; ++i) bits += this->treePositions[i].getEstimatedBits(this->biome, this->version);
		return bits;
    }
};

#endif