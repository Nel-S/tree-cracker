#ifndef __TEST_DATA_1_3_1_CUH
#define __TEST_DATA_1_3_1_CUH

#include "../Allowed Values for Settings.cuh"


// Worldseed: ??
// Structure seed: ??
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_3_1_1[] = {
    {
		static_cast<Version>(ExperimentalVersion::v1_6_4),
		TreeType::Oak,
		Coordinate(-14, 71),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_6_4),
		TreeType::Birch,
		Coordinate(-14, 63),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_6_4),
		TreeType::Oak,
		Coordinate(-20, 65),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_6_4),
		TreeType::Oak,
		Coordinate(-21, 62),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_6_4),
		TreeType::Oak,
		Coordinate(-9, 64),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
};

#endif