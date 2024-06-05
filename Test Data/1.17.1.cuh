#ifndef __TEST_DATA_1_17_1_CUH
#define __TEST_DATA_1_17_1_CUH

#include "../Allowed Values for Settings.cuh"

// Worldseed: -6973831680481511263
// Structure seed: 273817478412449
// Treechunk seed: ??
// VERIFIED
__device__ constexpr InputData TEST_DATA_17_1_1[] = {
    {
		Version::v1_17_1,
		TreeType::Oak,
		Coordinate(13, -41),
		Biome::Forest, // Should be Forest for Oak or Forest/Birch Forest for Birch.
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). Should be in the range [4,6] for Oak or [5,7] for Birch.

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
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
		Version::v1_17_1,
		TreeType::Oak,
		Coordinate(20, -45),
		Biome::Forest, // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		Version::v1_17_1, // (Should be 1.6.4- or 1.12.2+.)
		TreeType::Fancy_Oak,
		Coordinate(20, -48),
		Biome::Forest,       // (Should be Forest.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [3,14].)
	},
	{
		Version::v1_17_1,
		TreeType::Oak,
		Coordinate(20, -51),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(17, -55),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Oak,
		Coordinate(14, -53),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(23, -55),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
};


// Worldseed: -6973831680481511263
// Structure seed: 273817478412449
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_17_1_2[] = {
    {
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-856, 1472),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest), // Should be Forest for Oak or Forest/Birch Forest for Birch.
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). Should be in the range [4,6] for Oak or [5,7] for Birch.

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
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
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-855, 1467),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest), // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-858, 1466),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-861, 1463),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
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
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-853, 1462),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-854, 1459),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
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
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::Unknown  // Southeast
		})
	},
};


// Worldseed: -6973831680481511263
// Structure seed: 273817478412449
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_17_1_3[] = {
    {
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-856, 1472),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest), // Should be Forest for Oak or Forest/Birch Forest for Birch.
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). Should be in the range [4,6] for Oak or [5,7] for Birch.

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
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-855, 1467),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest), // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

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
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-858, 1466),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
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
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-861, 1463),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

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
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-853, 1462),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

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
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-854, 1459),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

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
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_17_1,
		TreeType::Birch,
		Coordinate(-860, 1457),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

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
			LeafState::Unknown  // Southeast
		})
	},
};

#endif