#ifndef __TEST_DATA_1_16_1_CUH
#define __TEST_DATA_1_16_1_CUH

#include "../Allowed Values for Settings.cuh"

// Worldseed: 3377920886278524622
// Structure seed: 221165750652622
// Treechunk seed: ??
__device__ constexpr InputData TEST_DATA_16_1_1[] = {
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

// Worldseed: -2891716383609216657
// Structure seed: 157527116063087
// Treechunk seed: ??
// From Chaos4669
__device__ constexpr InputData TEST_DATA_16_1_2[] = {
	{
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-170, 133),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Highest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-171, 129),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
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
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-167, 143),
		Biome::Forest, // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
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
			LeafState::Unknown  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Oak,
		Coordinate(-161, 132),
		Biome::Forest, // Should be Forest for Oak or Forest/Birch Forest for Birch.
		PossibleHeightsRange(), // The height of the tree's trunk (i.e. the number of logs it has). Should be in the range [4,6] for Oak or [5,7] for Birch.

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
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
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-165, 140),
		Biome::Forest,    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
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

// Worldseed: 123456789
// Structure seed: 123456789
// Treechunk seed: ??
// TODO: UNVERIFIED
__device__ constexpr InputData TEST_DATA_16_1_3[] = {
	{
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-442, 375),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-442, 379),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::Unknown, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
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
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-438, 380),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::Unknown, // Northeast
			LeafState::Unknown, // Southeast
			// Highest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-435, 376),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasNotPlaced, // Northwest
			LeafState::Unknown, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-434, 383),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::Unknown, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
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
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-444, 383),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

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
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasPlaced, // Northeast
			LeafState::LeafWasPlaced  // Southeast
		})
	},
	{
		Version::v1_16_1,
		TreeType::Birch,
		Coordinate(-440, 387),
		static_cast<Biome>(ExperimentalBiome::Birch_Forest),    // (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			// Lowest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasNotPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced, // Southeast
			// Middle y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasPlaced, // Southeast
			// Highest y-level:
			LeafState::LeafWasPlaced, // Northwest
			LeafState::LeafWasPlaced, // Southwest
			LeafState::LeafWasNotPlaced, // Northeast
			LeafState::LeafWasNotPlaced  // Southeast
		})
	},
};

#endif