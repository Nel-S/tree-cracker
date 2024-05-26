/*
   These are settings that haven't been fully implemented yet.
   There are no guarantees related to these in any way whatsoever. These may be present in future versions, or they may not be; these may ultimately be fully implemented, or they may not be; these may do nothing if changed, or they may catastrophically break the program if changed; and so on. Unless you are helping to develop this program, I would recommend leaving these as is.
*/

#ifndef __EXPERIMENTAL_SETTINGS_CUH
#define __EXPERIMENTAL_SETTINGS_CUH

#include "Base Logic.cuh"

#define DEBUG false

enum class Device {
	CUDA, CPU, AUTO = CUDA
};
enum class ExperimentalBiome {
   Taiga = 2
};
enum class ExperimentalVersion {
   v1_6_4, v1_8_9, v1_12_2, v1_17_1 = 6
};
enum class ExperimentalTreeType {
   Pine = 4, Spruce
};
enum class PrintSetting {
   None, RandomAndTextOnly, All,
   AUTO = All
};

/* The device one would prefer the program to run on.
   Note that the program will run much, much slower on a CPU than with CUDA.*/
constexpr Device DEVICE = Device::CUDA;

/* Reinterprets coordinates as being displacements from wherever (0,0) represents.
   (E.g. setting one tree as (0,0), another as (3, -2), and another as (0, 5) specifies the second tree is 3 blocks West and 2 blocks South of the first one, and the third tree is 5 blocks North of the first one.
   If enabled, this will 
   WARNING: This will slow down one's search by a factor of 256.*/
constexpr bool RELATIVE_COORDINATES_MODE = false;

constexpr PrintSetting PRINT_WORLDSEEDS_SETTING = PrintSetting::All;


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