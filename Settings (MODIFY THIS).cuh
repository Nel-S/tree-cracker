#ifndef __SETTINGS_CUH
#define __SETTINGS_CUH

// To find what options are valid for the following settings, see the following file:
#include "AllowedValuesForSettings.cuh"

// The device one would prefer the program to run on.
// Note that the program will run much, much slower on a CPU than with CUDA.
// TODO: Not implemented yet
constexpr Device DEVICE = Device::CUDA;
// The number of workers (processes, threads, etc.) one wishes the program to use.
// constexpr uint64_t NUMBER_OF_WORKERS = 4294967296;
constexpr uint64_t NUMBER_OF_WORKERS = 4;
// If one is using CUDA, the number of workers (threads) to use per GPU block.
#if CUDA_IS_PRESENT
constexpr uint64_t WORKERS_PER_BLOCK = 256;
#endif
// The filepath one wishes to direct the output to. WARNING: If that file already exists, the old file will be overwritten.
// If set to NULL, the output will only be printed to stdout.
constexpr const char *OUTPUT_FILEPATH = "output.txt";
constexpr bool TIME_PROGRAM = false;
constexpr bool SILENT_MODE = false;

// If one desires, the cracker can be split into separate "partial runs" that each check only a portion of all possible seeds.
// This allows for 
constexpr uint64_t NUMBER_OF_PARTIAL_RUNS = 8;
constexpr uint64_t PARTIAL_RUN_TO_BEGIN_FROM = 1; // This counts as 1, 2, ..., NUMBER_OF_PARTIAL_RUNS.

// The maximum number of results to allow per run.
// If set to 0, the program will instead calculate a reasonable limit automatically, based on the input data.
constexpr uint64_t MAX_NUMBER_OF_RESULTS_PER_RUN = 0;

// Reinterprets coordinates as being displacements from wherever (0,0) represents.
// (E.g. setting one tree as (0,0), another as (3, -2), and another as (0, 5) specifies the second tree is 3 blocks West and 2 blocks South of the first one,
//  and the third tree is 5 blocks North of the first one.
// WARNING: This will slow down one's search by a factor of 256.
// TODO: Not implemented yet
constexpr bool RELATIVE_COORDINATES_MODE = false;


// This list will hold each tree one has data for, and their attributes.
// Add to, delete from, or otherwise modify this example to suit your own input data.
__device__ constexpr InputData INPUT_DATA[] = {
	/* ----- START OF AN EXAMPLE OAK/BIRCH TREE ----- */
	// {
	// 	Version::v1_16_4, // The Minecraft version the tree was generated under.
	// 	TreeType::Oak,    // The type of tree it is.
	// 	Coordinate(0, 0), // The x- and z-coordinate of the tree.
	// 	Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
	// 	PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

	// 	(LeafState[NUMBER_OF_LEAF_POSITIONS]){
	// 		/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
	// 		(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

	// 		|       |       | ?---? | ?---? |       |       |
	// 		|       |       | ----- | ----- |  ?-?  |   -   |
	// 		|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
	// 		|       |       | ----- | ----- |  ?-?  |   -   |
	// 		|       |       | ?---? | ?---? |       |       |
	// 		   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
	// 		Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
	// 		// The lowest y-level (y=n+2 above):
	// 		LeafState::Unknown, // The state of the leaf in the Northwest corner
	// 		LeafState::Unknown, // The state of the leaf in the Southwest corner
	// 		LeafState::Unknown, // The state of the leaf in the Northeast corner
	// 		LeafState::Unknown, // The state of the leaf in the Southeast corner
	// 		// The middle y-level (y=n+3 above):
	// 		LeafState::Unknown, // The state of the leaf in the Northwest corner
	// 		LeafState::Unknown, // The state of the leaf in the Southwest corner
	// 		LeafState::Unknown, // The state of the leaf in the Northeast corner
	// 		LeafState::Unknown, // The state of the leaf in the Southeast corner
	// 		// The highest y-level (y=n+4 above):
	// 		LeafState::Unknown, // The state of the leaf in the Northwest corner
	// 		LeafState::Unknown, // The state of the leaf in the Southwest corner
	// 		LeafState::Unknown, // The state of the leaf in the Northeast corner
	// 		LeafState::Unknown  // The state of the leaf in the Southeast corner
	// 	}
	// },
	/* ----- END OF AN EXAMPLE OAK/BIRCH TREE ----- */
	
	// /* ----- START OF AN EXAMPLE LARGE OAK TREE ----- */
	// {
	// 	Version::v1_16_4,    // The Minecraft version the tree was generated under. (Should be 1.6.4 or below, or 1.12.2 or above.)
	// 	TreeType::Large_Oak, // The type of tree it is.
	// 	Coordinate(0, 0),    // The x- and z-coordinate of the tree.
	// 	Biome::Forest,       // The biome the coordinate above is within. (Should be Forest.)
	// 	PossibleHeightsRange(0), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [3,14].)
	// },
	// /* ----- END OF AN EXAMPLE OAK/BIRCH TREE ----- */

	// /* ----- START OF AN EXAMPLE PINE TREE (Tall with only one set of leaves at very top) ----- */
	// {
	// 	Version::v1_16_4, // The Minecraft version the tree was generated under.
	// 	TreeType::Pine,   // The type of tree it is.
	// 	Coordinate(0, 0), // The x- and z-coordinate of the tree.
	// 	Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
	// 	PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
	// 	PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
	// 	PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	// },
	// /* ----- END OF AN EXAMPLE PINE TREE ----- */

	// /* ----- START OF AN EXAMPLE SPRUCE TREE (Shorter with multiple sets of leaves) ----- */
	// {
	// 	Version::v1_16_4, // The Minecraft version the tree was generated under.
	// 	TreeType::Spruce, // The type of tree it is.
	// 	Coordinate(0, 0), // The x- and z-coordinate of the tree.
	// 	Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
	// 	PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,9].)
	// 	PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
	// 	PossibleHeightsRange(1), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
	// 	PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
	// 	PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
	// },
	// /* ----- END OF AN EXAMPLE SPRUCE TREE ----- */

	// /* ----- START OF AN EXAMPLE UNKNOWN-TYPE TREE ----- */
	// {
	// 	Version::v1_16_4, // The Minecraft version the tree was generated under.
	// 	TreeType::Unknown, // The type of tree it is.
	// 	Coordinate(0, 0), // The x- and z-coordinate of the tree.
	// 	Biome::Taiga,     // The biome the coordinate above is within.
	// 	PossibleHeightsRange(), // If known, the height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)", or in the worst-case scenario leave the range blank entirely.
	// },
	/* ----- END OF AN EXAMPLE UNKNOWN-TYPE TREE ----- */

	/* THE DATA FOR YOUR INPUT_DATA GO HERE -- MAKE SURE TO DELETE THE EXAMPLES ABOVE WHEN FINISHED */
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
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Southwest corner
			LeafState::Unknown, // The state of the leaf in the Northeast corner
			LeafState::Unknown, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::LeafWasNotPlaced, // The state of the leaf in the Northwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Southwest corner
			LeafState::LeafWasPlaced, // The state of the leaf in the Northeast corner
			LeafState::LeafWasNotPlaced  // The state of the leaf in the Southeast corner
		})
	},
};

#endif