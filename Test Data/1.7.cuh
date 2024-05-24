#ifndef __TEST_DATA_1_7_CUH
#define __TEST_DATA_1_7_CUH

#include "../Allowed Values for Settings.cuh"


// Worldseed: ??
// Structure seed: ??
// Treechunk seed: ??
// From Rehorted
__device__ constexpr InputData TEST_DATA_7_1[] = {
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(85, 149),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(1), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(1)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(76, 151),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(84, 143),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(9), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(1), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(1)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(76, 146),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(79, 136),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},

	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(-93, 336),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(3), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(-103, 338),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(4, 7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1, 2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange()    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(-101, 330),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(7, 8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(1, 2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Spruce),
		Coordinate(-89, 338),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(6, 7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(1, 2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(-92, 328),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(8, 9), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		TreeType::Unknown,
		Coordinate(-97, 329),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // The biome the coordinate above is within.
		PossibleHeightsRange(), // If known, the height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)", or in the worst-case scenario leave the range blank entirely.
	},

	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(-80, 353),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(-82, 358),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(7, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(-87, 359),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		static_cast<Version>(ExperimentalVersion::v1_8_9),
		static_cast<TreeType>(ExperimentalTreeType::Pine),
		Coordinate(-87, 351),
		static_cast<Biome>(ExperimentalBiome::Taiga),     // (Should be Taiga.)
		PossibleHeightsRange(10, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
};


// Seed: 123
// Structure seed: 123
// Treechunk seed: ??
// __device__ constexpr InputData TEST_DATA_11[] = {

// };


// Seed: ??
// Structure seed: ??
// Treechunk seed: 137138837835894(?)
// __device__ constexpr InputData TEST_DATA_12[] = {

// };


// "Some random map from Minecraft: Story Mode", seed: 2234065947811606375
// Structure seed: 280532635840359
// Treechunk seed: ??
// __device__ constexpr InputData TEST_DATA_13[] = {

// };

#endif