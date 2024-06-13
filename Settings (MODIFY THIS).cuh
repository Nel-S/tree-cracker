#ifndef __SETTINGS_CUH
#define __SETTINGS_CUH

/* To find what options are valid for the following settings, see the following file:*/
#include "Allowed Values for Settings.cuh"


// The number of workers (processes, threads, etc.) you wish the program to use.
constexpr const uint64_t NUMBER_OF_WORKERS = 4294967296;
// If you are using CUDA, the number of workers (threads) you wish to use per GPU block.
#if CUDA_IS_PRESENT
constexpr const uint64_t WORKERS_PER_BLOCK = 256;
#endif
/* The filepath you wish to direct the output to. If that file already exists, a file with a different name will be created instead.
   If set to NULL or the empty string, the output will only be printed to the screen.*/
constexpr const char *OUTPUT_FILEPATH = "output.txt";
/* How frequently you wish the program to print its statistics mid-run, like speed and estimated remaining time. (More specifically, it will print after every Nth dispatch of workers.)
   If set to 0, the statistics will never be printed.*/
constexpr const uint32_t PRINT_TIMESTAMPS_FREQUENCY = 256;
/* Whether you wish the program to run completely silently (no warnings, errors, timestamps, results, or anything else will be printed to the screen).
   This is most useful if e.g. the program is running on a cluster that does not support stdout.*/
constexpr const bool SILENT_MODE = false;
/* If you desire, the cracker can be split into separate "partial runs" that each check only a portion of all possible seeds.
   You can use this to resume your progress later if it is interrupted.*/
constexpr const uint64_t NUMBER_OF_PARTIAL_RUNS = 16;
constexpr const uint64_t PARTIAL_RUN_TO_BEGIN_FROM = 1; // This counts as 1, 2, ..., NUMBER_OF_PARTIAL_RUNS.
/* The maximum number of results to allow per run.
   If set to AUTO, the program will try to calculate a reasonable limit automatically, based on your input data.*/
constexpr const uint64_t MAX_NUMBER_OF_RESULTS_PER_RUN = AUTO;
/* Which types of values should be outputted.*/
constexpr OutputType TYPES_TO_OUTPUT = OutputType::Structure_Seeds;



/* This list holds each tree one has data for, and their attributes.
   Add to, delete from, or otherwise modify this example to suit your own input data.*/
__device__ constexpr InputData INPUT_DATA[] = {
	/* ----- START OF AN EXAMPLE OAK/BIRCH TREE ----- */
	{
		Version::v1_16_4, // The Minecraft version the tree was generated under.
		TreeType::Oak,    // The type of tree it is.
		Coordinate(0, 0), // The x- and z-coordinate of the tree.
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
			LeafState::Unknown, // The state of the leaf in the Northeast corner
			LeafState::Unknown, // The state of the leaf in the Southeast corner
			// The middle y-level (y=n+3 above):
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::Unknown, // The state of the leaf in the Southwest corner
			LeafState::Unknown, // The state of the leaf in the Northeast corner
			LeafState::Unknown, // The state of the leaf in the Southeast corner
			// The highest y-level (y=n+4 above):
			LeafState::Unknown, // The state of the leaf in the Northwest corner
			LeafState::Unknown, // The state of the leaf in the Southwest corner
			LeafState::Unknown, // The state of the leaf in the Northeast corner
			LeafState::Unknown  // The state of the leaf in the Southeast corner
		})
	},
	/* ----- END OF AN EXAMPLE OAK/BIRCH TREE ----- */
	
	/* ----- START OF AN EXAMPLE FANCY OAK TREE (Possibly large and overgrown; possibly has a more spherical leaf shape; or otherwise doesn't have the shape of a normal Oak tree) ----- */
	{
		Version::v1_16_1,    // The Minecraft version the tree was generated under. (Should be 1.6.4 or below, or 1.12.2 or above.)
		TreeType::Fancy_Oak, // The type of tree it is.
		Coordinate(0, 0),    // The x- and z-coordinate of the tree.
		Biome::Forest,       // The biome the coordinate above is within. (Should be Forest.)
		PossibleHeightsRange(3), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [3,14].)
	},
	/* ----- END OF AN EXAMPLE OAK/BIRCH TREE ----- */

	/* ----- START OF AN EXAMPLE UNKNOWN-TYPE TREE ----- */
	{
		Version::v1_16_4, // The Minecraft version the tree was generated under.
		TreeType::Unknown, // The type of tree it is.
		Coordinate(0, 0), // The x- and z-coordinate of the tree.
		Biome::Forest,     // The biome the coordinate above is within.
		PossibleHeightsRange(), // If known, the height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)", or in the worst-case scenario leave the range blank entirely.
	},
	/* ----- END OF AN EXAMPLE UNKNOWN-TYPE TREE ----- */

	/* THE DATA FOR YOUR INPUT_DATA GO HERE -- MAKE SURE TO DELETE THE EXAMPLES ABOVE WHEN FINISHED */





};


/* Alternatively, you can instead run one of the example input datasets, which can be seen in the following files: */
#include "Test Data/1.3.1.cuh"
#include "Test Data/1.6.4.cuh"
#include "Test Data/1.7.cuh"
#include "Test Data/1.8.9.cuh"
#include "Test Data/1.12.2.cuh"
#include "Test Data/1.14.4.cuh"
#include "Test Data/1.16.1.cuh"
#include "Test Data/1.16.4.cuh"
#include "Test Data/1.17.1.cuh"
/* To do that, comment out all of the INPUT_DATA code above, and uncomment and modify the line below.*/
// #define INPUT_DATA TEST_DATA_14_4_1



/* ===================================================================================================================
   THESE SETTINGS ARE EXPERIMENTAL AND STILL UNDER DEVELOPMENT.
	IF CHANGED FROM THEIR DEFAULTS, THERE ARE NO GUARANTEES REGARDING WHETHER THEY'LL WORK, WHETHER THEY'LL EVER WORK,
	WHETHER THEY'LL OUTRIGHT BREAK THE ENTIRE PROGRAM, ETC.
   =================================================================================================================== */

/* The device one would prefer the program to run on.
   Note that the program will run much, much slower on a CPU than with CUDA, and depending on the quality of one's input data may take literal months or years to finish.*/
constexpr Device DEVICE = Device::CUDA;

/* Reinterprets coordinates as being displacements from wherever (0,0) represents.
   (E.g. setting one tree as (0,0), another as (3, -2), and another as (0, 5) specifies the second tree is 3 blocks West and 2 blocks South of the first one, and the third tree is 5 blocks North of the first one.
   If enabled, this will 
   WARNING: This will slow down one's search by a factor of 256.*/
constexpr bool RELATIVE_COORDINATES_MODE = false;

/* Whether the game should test normal-biomes worldseeds, Large Biomes worldseeds, or both.*/
constexpr LargeBiomesFlag LARGE_BIOMES_WORLDSEEDS = LargeBiomesFlag::AUTO;


// /* ----- START OF AN EXAMPLE PINE TREE (Tall with only one set of leaves at very top) ----- */
// 	{
// 		Version::v1_14_4, // The Minecraft version the tree was generated under.
// 		static_cast<TreeType>(ExperimentalTreeType::Pine),   // The type of tree it is.
// 		Coordinate(0, 0), // The x- and z-coordinate of the tree.
// 		static_cast<Biome>(ExperimentalBiome::Taiga),     // The biome the coordinate above is within. (Should be Taiga.)
// 		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
// 		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
// 		PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
// 	},
// 	/* ----- END OF AN EXAMPLE PINE TREE ----- */

// 	/* ----- START OF AN EXAMPLE SPRUCE TREE (Shorter with multiple sets of leaves) ----- */
// 	{
// 		Version::v1_16_1, // The Minecraft version the tree was generated under.
// 		static_cast<TreeType>(ExperimentalTreeType::Spruce), // The type of tree it is.
// 		Coordinate(0, 0), // The x- and z-coordinate of the tree.
// 		static_cast<Biome>(ExperimentalBiome::Taiga),     // The biome the coordinate above is within. (Should be Taiga.)
// 		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,9].)
// 		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
// 		PossibleHeightsRange(1), // The number of layers of leaves directly above the top of the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
// 		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
// 		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
// 	},
// 	/* ----- END OF AN EXAMPLE SPRUCE TREE ----- */

#endif