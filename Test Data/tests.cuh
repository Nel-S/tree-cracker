#ifndef __TESTS_CUH
#define __TESTS_CUH

#include "../AllowedValuesForSettings.cuh"

// Seed: 3377920886278524622
__device__ constexpr InputData TEST_DATA_1[] = {
    {
		Version::v1_16_1, // The Minecraft version the tree was generated under.
		TreeType::Birch,    // The type of tree it is.
		Coordinate(-124, -22), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_16_1, // The Minecraft version the tree was generated under.
		TreeType::Oak,    // The type of tree it is.
		Coordinate(-125, -26), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::Unknown, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::Unknown, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_16_1, // The Minecraft version the tree was generated under.
		TreeType::Oak,    // The type of tree it is.
		Coordinate(-123, -29), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::Unknown, // The state of the leaf in the Southwest corner
			LeafState::Unknown, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_16_1, // The Minecraft version the tree was generated under.
		TreeType::Birch,    // The type of tree it is.
		Coordinate(-115, -27), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::Unknown, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::Unknown, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced  // The state of the leaf in the Southeast corner
		})
	},
};


// Seed: -3453423927651724977
__device__ constexpr InputData TEST_DATA_2[] = {
    {
		Version::v1_6_4, // The Minecraft version the tree was generated under.
		TreeType::Oak,    // The type of tree it is.
		Coordinate(1605, 381), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_6_4, // The Minecraft version the tree was generated under.
		TreeType::Birch,    // The type of tree it is.
		Coordinate(1601, 383), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::Unknown, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::Unknown, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_6_4,    // The Minecraft version the tree was generated under. (Should be 1.6.4 or below, or 1.12.2 or above.)
		TreeType::Large_Oak, // The type of tree it is.
		Coordinate(1596, 387),    // The x- and z-coordinate of the tree.
		Biome::Forest,       // The biome the coordinate above is within. (Should be Forest.)
		PossibleHeightsRange(), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [3,14].)
			/* Height was unknown, so left blank */
	},
};


// 1.3.1, seed: ??
__device__ constexpr InputData TEST_DATA_3[] = {
    {
		Version::v1_6_4, // The Minecraft version the tree was generated under.
		TreeType::Oak,    // The type of tree it is.
		Coordinate(-14, 71), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_6_4, // The Minecraft version the tree was generated under.
		TreeType::Birch,    // The type of tree it is.
		Coordinate(-14, 63), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_6_4, // The Minecraft version the tree was generated under.
		TreeType::Oak,    // The type of tree it is.
		Coordinate(-20, 65), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_6_4, // The Minecraft version the tree was generated under.
		TreeType::Oak,    // The type of tree it is.
		Coordinate(-21, 62), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::Unknown, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::Unknown, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced  // The state of the leaf in the Southeast corner
		})
	},
	{
		Version::v1_6_4, // The Minecraft version the tree was generated under.
		TreeType::Oak,    // The type of tree it is.
		Coordinate(-9, 64), // The x- and z-coordinate of the tree.
		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

			|       |       | ?---? | ?---? |       |       |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
			|       |       | ----- | ----- |  ?-?  |   -   |
			|       |       | ?---? | ?---? |       |       |
			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
			// The lowest y-level (y=n+2 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced  // The state of the leaf in the Southeast corner
		})
	},
};


// Seed: ??
__device__ constexpr InputData TEST_DATA_4[] = {
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(553, 4879), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(3), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(556, 4884), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(564, 4873), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(2), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(565, 4876), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(561, 4880), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
};


// 1.7, seed: ??
// From Rehorted
__device__ constexpr InputData TEST_DATA_5[] = {
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(85, 149), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(8), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(1), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange(1)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(76, 151), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(2), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(84, 143), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(9), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(1), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange(1)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(76, 146), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(10), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(79, 136), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(7), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},

	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(-93, 336), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(5), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(3), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(-103, 338), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(4, 7), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(1, 2), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange()    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(-101, 330), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(7, 8), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(1, 2), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Spruce, // The type of tree it is.
		Coordinate(-89, 338), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(6, 7), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". Supports ranges if there is uncertainty. (Should be in the range [4,9].)
		PossibleHeightsRange(2), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
		PossibleHeightsRange(1, 2), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(-92, 328), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(8, 9), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Unknown, // The type of tree it is.
		Coordinate(-97, 329), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within.
		PossibleHeightsRange(), // If known, the height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)", or in the worst-case scenario leave the range blank entirely.
	},

	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(-80, 353), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(2, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(-82, 358), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(7, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(-87, 359), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(6, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(2), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
	{
		Version::v1_8_9, // The Minecraft version the tree was generated under.
		TreeType::Pine,   // The type of tree it is.
		Coordinate(-87, 351), // The x- and z-coordinate of the tree.
		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
		PossibleHeightsRange(10, PossibleHeightsRange::NO_MAXIMUM), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
		PossibleRadiiRange(5),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
		PossibleRadiiRange(3, PossibleRadiiRange::NO_MAXIMUM), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	},
};


// TODO: Convert from Andrew's input_data.cuh
// Seed: 123
// __device__ constexpr InputData TEST_DATA_7[] = {

// };


// Seed: 123
// __device__ constexpr InputData TEST_DATA_8[] = {

// };


// Seed: -5141540374460396599
// From Advent
// __device__ constexpr InputData TEST_DATA_9[] = {

// };


// Seed: 123
// __device__ constexpr InputData TEST_DATA_10[] = {

// };


// Seed: ??
// __device__ constexpr InputData TEST_DATA_11[] = {

// };


// "Some random map from Minecraft: Story Mode", seed: 2234065947811606375
// __device__ constexpr InputData TEST_DATA_12[] = {

// };


// Seed: -3453423927651724977
// __device__ constexpr InputData TEST_DATA_13[] = {

// };

#endif