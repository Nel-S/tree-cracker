#ifndef __TEST_DATA_1_12_2_CUH
#define __TEST_DATA_1_12_2_CUH

#include "../Allowed Values for Settings.cuh"

// Super Smash Bros. 7 background, seed: ??
constexpr InputData TEST_DATA_12_1_1[] = {
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(-715, 138),
		static_cast<Biome>(ExperimentalBiome::Taiga), // (Should be Taiga.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(-726, 146),
		static_cast<Biome>(ExperimentalBiome::Taiga), // (Should be Taiga.)
		PossibleHeightsRange(7, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		static_cast<TreeType>(ExperimentalTreeType::Spruce), // The type of tree it is.
		Coordinate(-719, 141),
		static_cast<Biome>(ExperimentalBiome::Taiga), // (Should be Taiga.)
		PossibleHeightsRange(7, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		static_cast<TreeType>(ExperimentalTreeType::Spruce), // The type of tree it is.
		Coordinate(-728, 151),
		static_cast<Biome>(ExperimentalBiome::Taiga), // (Should be Taiga.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(1)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},

	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		static_cast<TreeType>(ExperimentalTreeType::Spruce), // The type of tree it is.
		Coordinate(-698, 147),
		static_cast<Biome>(ExperimentalBiome::Taiga), // (Should be Taiga.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		static_cast<TreeType>(ExperimentalTreeType::Spruce), // The type of tree it is.
		Coordinate(-703, 145),
		static_cast<Biome>(ExperimentalBiome::Taiga), // (Should be Taiga.)
		PossibleHeightsRange(7, 8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(2, PossibleHeightsRange::NO_MAXIMUM), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(1)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		static_cast<TreeType>(ExperimentalTreeType::Spruce), // The type of tree it is.
		Coordinate(-710, 146),
		static_cast<Biome>(ExperimentalBiome::Taiga), // (Should be Taiga.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(-706, 142),
		static_cast<Biome>(ExperimentalBiome::Taiga), // (Should be Taiga.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},

	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		TreeType::Oak, 
		Coordinate(-693, 99),
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		TreeType::Oak, 
		Coordinate(-691, 95),
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		TreeType::Oak, 
		Coordinate(-684, 95),
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_12_2),
		TreeType::Birch, 
		Coordinate(-686, 97),
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
};

#endif