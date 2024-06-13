#ifndef __TEST_DATA_1_6_4_CUH
#define __TEST_DATA_1_6_4_CUH

#include "../Allowed Values for Settings.cuh"


// Worldseed: -3453423927651724977
// Structure seed: 274036588024143
// Treechunk seed: ??
// Population chunk call #: 3759
__device__ constexpr InputData TEST_DATA_6_4_1[] = {
    {
		static_cast<Version>(ExperimentalVersion::v1_6_4),
		TreeType::Oak,
		Coordinate(1605, 381),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_6_4),
		TreeType::Birch,
		Coordinate(1601, 383),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_6_4), // (Should be 1.6.4- or 1.12.2+.)
		TreeType::Fancy_Oak,
		Coordinate(1596, 387),
		Biome::Forest,       // (Should be Forest.)
		PossibleHeightsRange(), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [3,14].)
			/* Height was unknown, so left blank */
	},
};

#endif