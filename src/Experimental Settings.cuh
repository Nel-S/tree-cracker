/*
   Settings that haven't been fully implemented yet.
   There are no guarantees about what messing with these will do to the program at large, so I recommend leaving them as is.
*/

#ifndef __EXPERIMENTAL_SETTINGS_CUH
#define __EXPERIMENTAL_SETTINGS_CUH

#include "../Settings (MODIFY THIS).cuh"

enum class ExperimenalDevice {
   CPU = 1
};

// The device one would prefer the program to run on.
// Note that the program will run much, much slower on a CPU than with CUDA.
// TODO: Not implemented yet
constexpr Device DEVICE = Device::CUDA;

// Reinterprets coordinates as being displacements from wherever (0,0) represents.
// (E.g. setting one tree as (0,0), another as (3, -2), and another as (0, 5) specifies the second tree is 3 blocks West and 2 blocks South of the first one,
//  and the third tree is 5 blocks North of the first one.
// WARNING: This will slow down one's search by a factor of 256.
// TODO: Not implemented yet
constexpr bool RELATIVE_COORDINATES_MODE = false;

#endif