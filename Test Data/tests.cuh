#ifndef __TESTS_CUH
#define __TESTS_CUH

#include "../Allowed Values for Settings.cuh"

// Confirmed to work:
// 1.14.4 forest, oak/birch (TEST_DATA_7)
// 1.16.1 forest, oak/birch (TEST_DATA_1)

// Confirmed not to work (currently):
// 1.6.4 forest, oak/birch/large oak (TEST_DATA_2)

// Worldseed: 3377920886278524622
// Structure seed: 221165750652622
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_1[] = {
    {
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-124, -22),
		Biome::Forest, // Should be Forest for Oak or Forest/Birch Forest for Birch.
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). Should be in the range [4,6] for Oak or [5,7] for Birch.

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Oak,
		Coordinate(-125, -26),
		Biome::Forest, // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Oak,
		Coordinate(-123, -29),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-115, -27),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
};


// Worldseed: -3453423927651724977
// Structure seed: 274036588024143
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_2[] = {
    {
		Version::v1_6_4,
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
		Version::v1_6_4,
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
		Version::v1_6_4, // (Should be 1.6.4- or 1.12.2+.)
		TreeType::Large_Oak,
		Coordinate(1596, 387),
		Biome::Forest,       // (Should be Forest.)
		PossibleHeightsRange(), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [3,14].)
			/* Height was unknown, so left blank */
	},
};


// 1.3.1, worldseed: ??
// Structure seed: ??
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_3[] = {
    {
		Version::v1_6_4,
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
		Version::v1_6_4,
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
		Version::v1_6_4,
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
		Version::v1_6_4,
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
		Version::v1_6_4,
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


// Worldseed: ??
// Structure seed: ??
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_4[] = {
	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(553, 4879),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(3), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(556, 4884),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(564, 4873),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(565, 4876),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(561, 4880),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
};


// 1.7, worldseed: ??
// Structure seed: ??
// Treechunk seed: ??
// From Rehorted
__device__ constexpr InputData TEST_DATA_5[] = {
	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(85, 149),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(1), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(1)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(76, 151),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(84, 143),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(9), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(1), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(1)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(76, 146),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(79, 136),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},

	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(-93, 336),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(3), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(-103, 338),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(4, 7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(1, 2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange()    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(-101, 330),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(7, 8), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(1, 2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Spruce,
		Coordinate(-89, 338),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(6, 7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. (Should be in the range [1,2].)
		PossibleHeightsRange(1, 2), // The number of layers of leaves above the trunk. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(-92, 328),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(8, 9), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9,
		TreeType::Unknown,
		Coordinate(-97, 329),
		Biome::Taiga,     // The biome the coordinate above is within.
		PossibleHeightsRange(), // If known, the height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)", or in the worst-case scenario leave the range blank entirely.
	},

	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(-80, 353),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(-82, 358),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(7, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(-87, 359),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(2), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9,
		TreeType::Pine,
		Coordinate(-87, 351),
		Biome::Taiga,     // (Should be Taiga.)
		PossibleHeightsRange(10, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. (Should be in the range [4,5].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. (Should be in the range [1,3].)
	},
};


// Seed: 123
// Structure seed: 123
// Treechunk seed: 13108863711061
__device__ constexpr InputData TEST_DATA_7[] = {
	{
		Version::v1_14_4,
		TreeType::Oak,
		Coordinate(-48, 99),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_14_4,
		TreeType::Birch,
		Coordinate(-47, 96),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_14_4,
		TreeType::Birch,
		Coordinate(-44, 98),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		Version::v1_14_4,
		TreeType::Oak,
		Coordinate(-46, 102),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_14_4,
		TreeType::Oak,
		Coordinate(-43, 101),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_14_4,
		TreeType::Oak,
		Coordinate(-34, 100),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		Version::v1_14_4,
		TreeType::Oak,
		Coordinate(-37, 106),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_14_4,
		TreeType::Oak,
		Coordinate(-43, 110),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
};


// TODO: Convert from Andrew's input_data.cuh
// Seed: 123
// Structure seed: 123
// Treechunk seed: 241689593949439
// __device__ constexpr InputData TEST_DATA_8[] = {

// };


// Seed: -5141540374460396599
// Structure seed: 163025113156553
// Treechunk seeds: 69261861613140, 83751666894233
// From Advent
// __device__ constexpr InputData TEST_DATA_9[] = {

// };


// Seed: 123
// Structure seed: 123
// Treechunk seed: ??
// __device__ constexpr InputData TEST_DATA_10[] = {

// };


// Seed: ??
// Structure seed: ??
// Treechunk seed: 137138837835894(?)
// __device__ constexpr InputData TEST_DATA_11[] = {

// };


// "Some random map from Minecraft: Story Mode", seed: 2234065947811606375
// Structure seed: 280532635840359
// Treechunk seed: ??
// __device__ constexpr InputData TEST_DATA_12[] = {

// };

#endif