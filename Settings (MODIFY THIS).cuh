#ifndef __SETTINGS_CUH
#define __SETTINGS_CUH

// To find what options are valid for the following settings, see the following file:
#include "Allowed Values for Settings.cuh"
#include "Test Data/Tests.cuh"


// The number of workers (processes, threads, etc.) you wish the program to use.
constexpr uint64_t NUMBER_OF_WORKERS = 4294967296;
// If you are using CUDA, the number of workers (threads) you wish to use per GPU block.
#if CUDA_IS_PRESENT
constexpr uint64_t WORKERS_PER_BLOCK = 256;
#endif
/* The filepath you wish to direct the output to. If that file already exists, a file with a different name will be created instead.
   If set to NULL, the output will only be printed to the screen.*/
constexpr const char *OUTPUT_FILEPATH = "output.txt";
// Whether you wish the program to print its runtime. 
constexpr bool TIME_PROGRAM = true;
/* Whether you wish the program to run completely silently (no warnings, errors, results, or anything else will be printed to the screen).
   This is most useful if e.g. the program is running on a cluster that does not support stdout.*/
constexpr bool SILENT_MODE = false;
/* If you desire, the cracker can be split into separate "partial runs" that each check only a portion of all possible seeds.
   You can use this to (at least partially) resume your progress later if it is interrupted.*/
constexpr uint64_t NUMBER_OF_PARTIAL_RUNS = 8;
constexpr uint64_t PARTIAL_RUN_TO_BEGIN_FROM = 1; // This counts as 1, 2, ..., NUMBER_OF_PARTIAL_RUNS.
/* The maximum number of results to allow per run.
   If set to AUTO, the program will try to calculate a reasonable limit automatically, based on your input data.*/
// constexpr uint64_t MAX_NUMBER_OF_RESULTS_PER_RUN = AUTO;
constexpr uint64_t MAX_NUMBER_OF_RESULTS_PER_RUN = AUTO;


/* This list holds each tree one has data for, and their attributes.
   Add to, delete from, or otherwise modify this example to suit your own input data.*/
#define INPUT_DATA TEST_DATA_2
// __device__ constexpr InputData INPUT_DATA[] = {
// 	/* ----- START OF AN EXAMPLE OAK/BIRCH TREE ----- */
// 	{
// 		Version::v1_16_4, // The Minecraft version the tree was generated under.
// 		TreeType::Oak,    // The type of tree it is.
// 		Coordinate(0, 0), // The x- and z-coordinate of the tree.
// 		Biome::Forest,    // The biome the coordinate above is within. (Should be Forest for Oak or Forest/Birch Forest for Birch.)
// 		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,6] for Oak or [5,7] for Birch.)

// 		std::array<LeafState, NUMBER_OF_LEAF_POSITIONS>({
// 			/* When generating oak or birch trees, the game will (pseudo)randomly keep, or reset to air, 12 exact leaves at the corners of the tree.
// 			(Below is a layer-by-layer diagram of a tree; the ?s are where the corner leaves are kept/removed.):

// 			|       |       | ?---? | ?---? |       |       |
// 			|       |       | ----- | ----- |  ?-?  |   -   |
// 			|   @   |   @   | --@-- | --@-- |  -@-  |  ---  |
// 			|       |       | ----- | ----- |  ?-?  |   -   |
// 			|       |       | ?---? | ?---? |       |       |
// 			   y=n    y=n+1   y=n+2   y=n+3   y=n+4   y=n+5
			
// 			Whether each of these leaves was placed or not yields information about the original worldseed.*/
			
// 			// The lowest y-level (y=n+2 above):
// 			LeafState::Unknown, // The state of the leaf in the Northwest corner
// 			LeafState::Unknown, // The state of the leaf in the Southwest corner
// 			LeafState::Unknown, // The state of the leaf in the Northeast corner
// 			LeafState::Unknown, // The state of the leaf in the Southeast corner
// 			// The middle y-level (y=n+3 above):
// 			LeafState::Unknown, // The state of the leaf in the Northwest corner
// 			LeafState::Unknown, // The state of the leaf in the Southwest corner
// 			LeafState::Unknown, // The state of the leaf in the Northeast corner
// 			LeafState::Unknown, // The state of the leaf in the Southeast corner
// 			// The highest y-level (y=n+4 above):
// 			LeafState::Unknown, // The state of the leaf in the Northwest corner
// 			LeafState::Unknown, // The state of the leaf in the Southwest corner
// 			LeafState::Unknown, // The state of the leaf in the Northeast corner
// 			LeafState::Unknown  // The state of the leaf in the Southeast corner
// 		})
// 	},
// 	/* ----- END OF AN EXAMPLE OAK/BIRCH TREE ----- */
	
// 	/* ----- START OF AN EXAMPLE LARGE OAK TREE ----- */
// 	{
// 		Version::v1_16_4,    // The Minecraft version the tree was generated under. (Should be 1.6.4 or below, or 1.12.2 or above.)
// 		TreeType::Large_Oak, // The type of tree it is.
// 		Coordinate(0, 0),    // The x- and z-coordinate of the tree.
// 		Biome::Forest,       // The biome the coordinate above is within. (Should be Forest.)
// 		PossibleHeightsRange(0), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [3,14].)
// 	},
// 	/* ----- END OF AN EXAMPLE OAK/BIRCH TREE ----- */

// 	/* ----- START OF AN EXAMPLE PINE TREE (Tall with only one set of leaves at very top) ----- */
// 	{
// 		Version::v1_16_4, // The Minecraft version the tree was generated under.
// 		TreeType::Pine,   // The type of tree it is.
// 		Coordinate(0, 0), // The x- and z-coordinate of the tree.
// 		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
// 		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
// 		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
// 		PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
// 	},
// 	/* ----- END OF AN EXAMPLE PINE TREE ----- */

// 	/* ----- START OF AN EXAMPLE SPRUCE TREE (Shorter with multiple sets of leaves) ----- */
// 	{
// 		Version::v1_16_4, // The Minecraft version the tree was generated under.
// 		TreeType::Spruce, // The type of tree it is.
// 		Coordinate(0, 0), // The x- and z-coordinate of the tree.
// 		Biome::Taiga,     // The biome the coordinate above is within. (Should be Taiga.)
// 		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,9].)
// 		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
// 		PossibleHeightsRange(1), // The number of layers of leaves above the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
// 		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
// 		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
// 	},
// 	/* ----- END OF AN EXAMPLE SPRUCE TREE ----- */

// 	/* ----- START OF AN EXAMPLE UNKNOWN-TYPE TREE ----- */
// 	{
// 		Version::v1_16_4, // The Minecraft version the tree was generated under.
// 		TreeType::Unknown, // The type of tree it is.
// 		Coordinate(0, 0), // The x- and z-coordinate of the tree.
// 		Biome::Taiga,     // The biome the coordinate above is within.
// 		PossibleHeightsRange(), // If known, the height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)", or in the worst-case scenario leave the range blank entirely.
// 	},
// 	/* ----- END OF AN EXAMPLE UNKNOWN-TYPE TREE ----- */

// 	/* THE DATA FOR YOUR INPUT_DATA GO HERE -- MAKE SURE TO DELETE THE EXAMPLES ABOVE WHEN FINISHED */





// };

#endif