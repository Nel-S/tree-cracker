#ifndef __TEST_DATA_1_8_9_CUH
#define __TEST_DATA_1_8_9_CUH

#include "../Allowed Values for Settings.cuh"

// Worldseed: ??
// Structure seed: ??
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_8_9_1[] = {
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(553, 4879),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(3), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(556, 4884),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(564, 4873),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(565, 4876),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(561, 4880),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
};


// Seed: 123
// Structure seed: 123
// Treechunk seed: 241689593949439
__device__ constexpr InputData TEST_DATA_8_9_2[] = {
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(133, 493),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(129, 501),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(135, 499),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(121, 499),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(122, 492),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Birch,
		Coordinate(123, 489),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Birch,
		Coordinate(132, 503),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
};


// Seed: -5141540374460396599
// Structure seed: 163025113156553
// Treechunk seeds: 69261861613140, 83751666894233
// From Advent
__device__ constexpr InputData TEST_DATA_8_9_3[] = {
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Birch,
		Coordinate(267, 175),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Birch,
		Coordinate(268, 183),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Birch,
		Coordinate(274, 169),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(267, 180),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(277, 182),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(279, 173),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Oak,
		Coordinate(279, 168),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	
	//TODO: Finish
};

#endif