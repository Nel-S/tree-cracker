#ifndef __ALLOWED_VALUES_FOR_SETTINGS_CUH
#define __ALLOWED_VALUES_FOR_SETTINGS_CUH

#include "src/Base.cuh"
#include <array>

// The list of supported devices.
enum class Device {
	CPU, CUDA
};

// The list of supported versions.
enum class Version {
	// Beta_1_7,
	// Beta_1_8,
	// v1_0,   v1_0_0 = v1_0,
	// v1_1,   v1_1_0 = v1_1,
	// v1_2,   v1_2_0 = v1_2,  v1_2_1,  v1_2_2,  v1_2_3,  v1_2_4,  v1_2_5,
	// v1_3,   v1_3_0 = v1_3,  v1_3_1,  v1_3_2,
	// v1_4,   v1_4_0 = v1_4,  v1_4_1,  v1_4_2,  v1_4_3,  v1_4_4,  v1_4_5,  v1_4_6,  v1_4_7,
	// v1_5,   v1_5_0 = v1_5,  v1_5_1,  v1_5_2,
	/* v1_6,   v1_6_0 = v1_6,  v1_6_1,  v1_6_2,  v1_6_3, */  v1_6_4,
	// v1_7,   v1_7_0 = v1_7,  v1_7_1,  v1_7_2,  v1_7_3,  v1_7_4,  v1_7_5,  v1_7_6,  v1_7_7, v1_7_8, v1_7_9, v1_7_10,
	/* v1_8,   v1_8_0 = v1_8,  v1_8_1,  v1_8_2,  v1_8_3,  v1_8_4,  v1_8_5,  v1_8_6,  v1_8_7, v1_8_8, */ v1_8_9,
	// v1_9,   v1_9_0 = v1_9,  v1_9_1,  v1_9_2,  v1_9_3,  v1_9_4,
	// v1_10, v1_10_0 = v1_10, v1_10_1, v1_10_2,
	// v1_11, v1_11_0 = v1_11, v1_11_1, v1_11_2,
	/* v1_12, v1_12_0 = v1_12, v1_12_1, */ v1_12_2,
	// v1_13, v1_13_0 = v1_13, v1_13_1, v1_13_2,
	/* v1_14, v1_14_0 = v1_14, v1_14_1, v1_14_2, v1_14_3, */ v1_14_4,
	// v1_15, v1_15_0 = v1_15, v1_15_1, v1_15_2,
	/* v1_16, v1_16_0 = v1_16, */ v1_16_1, /* v1_16_2, v1_16_3, */ v1_16_4, /* v1_16_5, */
	// v1_17, v1_17_0 = v1_17, v1_17_1,
		// NelS: Possibly unsupportable due to xoroshiro128++?
	// v1_18, v1_18_0 = v1_18, v1_18_1, v1_18_2,
	// v1_19, v1_19_0 = v1_19, v1_19_1, v1_19_2, v1_19_3, v1_19_4,
	// v1_20, v1_20_0 = v1_20, v1_20_1, v1_20_2, v1_20_3, v1_20_4, v1_20_5, v1_20_6,
	// Unknown
};

// The list of supported tree types.
enum class TreeType {
	Oak, Large_Oak, Spruce, Pine, Birch, Unknown
};

// The list of supported biomes.
enum class Biome {
	Forest, Birch_Forest, Taiga //, Unknown
};

// For Oak and Birch trees, a list of which corner leaf each entry in the LeafState array corresponds to (as well as tracking how many leaf positions in total there are).
enum class LeafPosition {
	NorthwestLowerLeaf, SouthwestLowerLeaf, NortheastLowerLeaf, SoutheastLowerLeaf, NorthwestMiddleLeaf, SouthwestMiddleLeaf, NortheastMiddleLeaf, SoutheastMiddleLeaf, NorthwestUpperLeaf, SouthwestUpperLeaf, NortheastUpperLeaf, SoutheastUpperLeaf, NumberOfLeafPositions
};
// (To save on typing)
#define NUMBER_OF_LEAF_POSITIONS static_cast<size_t>(LeafPosition::NumberOfLeafPositions)

// The list of supported corner leaf states.
enum class LeafState {
	LeafWasNotPlaced, LeafWasPlaced, Unknown
};

struct InputData {
	/* The overall layout of each input element.
	   Don't modify this--place one's input data in "Settings (MODIFY THIS).cuh" instead.*/
	Version version;
	TreeType treeType;
	Coordinate coordinate;
	Biome biome;
	PossibleHeightsRange trunkHeight;
	/* Only one of the following sets of attributes can be present for each tree.
	   (Large Oaks don't have any such attributes.)*/
	union {
		// Non-large Oak/Birch
		struct {
			LeafState leafStates[NUMBER_OF_LEAF_POSITIONS];
		};
		// Pine
		struct {
			PossibleHeightsRange leavesHeight;
			PossibleRadiiRange pineLeavesWidestRadius;
		};
		// Spruce
		struct {
			PossibleHeightsRange logsBelowBottommostLeaves, leavesAboveTrunk;
			PossibleRadiiRange spruceLeavesWidestRadius, topmostLeavesRadius;
		};
	};

	// Unknown tree/Generic constructor, trunkHeight unknown
	constexpr InputData(const Version version, const TreeType treeType, const Coordinate &coordinate, const Biome biome) : version(version), treeType(treeType), coordinate(coordinate), biome(biome),
		leafStates{LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown} {}
	// Large Oak tree/Generic constructor, trunkHeight known
	constexpr InputData(const Version version, const TreeType treeType, const Coordinate &coordinate, const Biome biome, const PossibleHeightsRange &trunkHeight) : version(version), treeType(treeType), coordinate(coordinate), biome(biome),
		trunkHeight(trunkHeight),
		leafStates{LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown, LeafState::Unknown} {}
	// Non-large Oak/Birch tree constructor
	// constexpr InputData(const Version version, const TreeType treeType, const Coordinate &coordinate, const Biome biome, const PossibleHeightsRange &trunkHeight, const LeafState leafStates[NUMBER_OF_LEAF_POSITIONS]) : version(version), treeType(treeType), coordinate(coordinate), biome(biome),
	// 	trunkHeight(trunkHeight),
	// 	// NelS: Surely there must be a cleaner way of doing this.
	// 	leafStates{leafStates[0], leafStates[1], leafStates[2], leafStates[3], leafStates[4], leafStates[5], leafStates[6], leafStates[7], leafStates[8], leafStates[9], leafStates[10], leafStates[11]} {};
	constexpr InputData(const Version version, const TreeType treeType, const Coordinate &coordinate, const Biome biome, const PossibleHeightsRange &trunkHeight, const std::array<LeafState, NUMBER_OF_LEAF_POSITIONS> leafStates) : version(version), treeType(treeType), coordinate(coordinate), biome(biome),
		trunkHeight(trunkHeight),
		// NelS: Surely there must be a cleaner way of doing this.
		leafStates{leafStates[0], leafStates[1], leafStates[2], leafStates[3], leafStates[4], leafStates[5], leafStates[6], leafStates[7], leafStates[8], leafStates[9], leafStates[10], leafStates[11]} {};
	// Pine tree constructor
	constexpr InputData(const Version version, const TreeType treeType, const Coordinate &coordinate, const Biome biome, const PossibleHeightsRange &trunkHeight, const PossibleHeightsRange &leavesHeight, const PossibleRadiiRange &leavesWidestRadius) : version(version), treeType(treeType), coordinate(coordinate), biome(biome),
		trunkHeight(trunkHeight),
		pineLeavesWidestRadius(leavesWidestRadius),
		leavesHeight(leavesHeight) {};
	// Spruce tree constructor
	constexpr InputData(const Version version, const TreeType treeType, const Coordinate &coordinate, const Biome biome, const PossibleHeightsRange &trunkHeight, const PossibleHeightsRange &logsBelowBottommostLeaves, const PossibleHeightsRange &leavesAboveTrunk, const PossibleRadiiRange &leavesWidestRadius, const PossibleRadiiRange &topmostLeavesRadius) : version(version), treeType(treeType), coordinate(coordinate), biome(biome),
		trunkHeight(trunkHeight),
		logsBelowBottommostLeaves(logsBelowBottommostLeaves),
		leavesAboveTrunk(leavesAboveTrunk),
		spruceLeavesWidestRadius(leavesWidestRadius),
		topmostLeavesRadius(topmostLeavesRadius) {};
};

#endif