#ifndef __FILTERS_CUH
#define __FILTERS_CUH

#include "ChunkRandom Reversal Logic.cuh"
#include "Biome Logic.cuh"
// #include <cstdio>

// TODO: Change to storageA and storageB which results flip-flop between, removing the need for transferResults
// (Though that would cause adding/removing any more filters to become a major headache...)
__device__ uint64_t filterInputs[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN];
__device__ uint32_t filter3_masks[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN/32];

__managed__ uint64_t filter1_numberOfResultsPerRun = 0;
__managed__ uint64_t filter2_numberOfResultsPerRun = 0;
__managed__ uint64_t filter3_numberOfResultsPerRun = 0;
__managed__ uint64_t treechunkFilter_numberOfResultsPerRun = 0;

__device__ Stage2Results stage2filterInputs[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN]; // To prevent new results from accidentally erasing the old inputs mid-filter
__device__ Stage2Results stage2filterResults[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN]; // To prevent new results from accidentally erasing the old inputs mid-filter
__managed__ uint64_t filter5_numberOfResultsPerRun = 0;
__managed__ uint64_t filter6_numberOfResultsPerRun = 0;
__managed__ uint64_t filter7_numberOfResultsPerRun = 0;
__device__ uint32_t filter5_masks[(ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN + 31) / 32];
__device__ uint32_t filter8_masks[(ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN + 31) / 32];



#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void transferResults() {
	uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
#else
void *transferResults(void *start) {
	ThreadData *star = static_cast<ThreadData *>(start);
#endif
	if (index >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	filterInputs[index] = filterResults[index];
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those that could generate the first tree in POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex]. Specifically, successful states
	- would choose the first tree's x-coordinate (if RELATIVE_COORDINATES_MODE is false);
	- on the next call, would choose the first tree's z-coordinate (if RELATIVE_COORDINATES_MODE is false);
	- on the next call(s), would choose the first tree's type; and
	- on subsequent calls, would choose all of the first tree's attributes.
   filter1_numberOfResultsPerRun must be set to 0 beforehand.
   Values are generated internally and outputted to filterResults[], with the final count being stored in filter1_numberOfResultsPerRun.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter1(const uint64_t start) {
	uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	uint64_t seed = (RELATIVE_COORDINATES_MODE ? UINT64_C(0) : (static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[0].populationChunkXOffset) << 44)) + start + index;
#else
void *filter1(void *start) {
	ThreadData *star = static_cast<ThreadData *>(start);
	uint64_t seed = (RELATIVE_COORDINATES_MODE ? UINT64_C(0) : (static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[0].populationChunkXOffset) << 44)) + star->start + star->index;
#endif

	Random treeRandom = Random::withSeed(seed);
	if (RELATIVE_COORDINATES_MODE) {
		if (!POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[0].testTypeAndAttributes(treeRandom, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) {
			#if CUDA_IS_PRESENT
				return;
			#else
				return NULL;
			#endif
		}
		seed = Random::withSeed(seed).skip<-2>().seed; // Step back two calls to account for x call (nextInt(16)) and z call (nextInt(16))
	} else { // Absolute coordinates mode
		if (!POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[0].testZTypeAndAttributes(treeRandom, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) {
			#if CUDA_IS_PRESENT
				return;
			#else
				return NULL;
			#endif
		}
	}
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter1_numberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those whose surrounding states, within the maximum possible range of calls, could generate all listed trees' positions and types. Specifically, successful states
	- pass filter1, and
	- for each other tree in the chunk, have a state between -.maxCalls and .maxCalls advancements away from the tested state that could generate the tree's
		- x-coordinate,
		- z-coordinate, and
		- type.
   filter2_numberOfResultsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in filter2_numberOfResultsPerRun.*/
// TODO: Unify with filter6
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter2() {
	uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
#else
void *filter2(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= filter1_numberOfResultsPerRun) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}

	uint64_t seed = filterInputs[index];

	uint32_t found = 0;
	Random random = Random::withSeed(seed);
	int32_t calls = POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls;
	if (calls & 256) random.skip<-256>();
	if (calls & 128) random.skip<-128>();
	if (calls & 64) random.skip<-64>();
	if (calls & 32) random.skip<-32>();
	if (calls & 16) random.skip<-16>();
	if (calls & 8) random.skip<-8>();
	if (calls & 4) random.skip<-4>();
	if (calls & 2) random.skip<-2>();
	if (calls & 1) random.skip<-1>();

	for (int32_t _currentCall = -POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; _currentCall <= POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; ++_currentCall) {
		// if (random.seed == 13108863711061) printf("\t(Encountered correct treechunk seed with seed %" PRIu64 ")\n", seed);
		#pragma unroll
		for (uint32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
			Random treeRandom(random);
			if (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZAndType(treeRandom, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
		}
		random.skip<1>();
	}

	// if (seed == 13108863711061) printf("\t(%" PRIu32 "\n", found);
	if (found != ((UINT32_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) - 1)) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter2_numberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those whose surrounding states, within the maximum possible range of calls, could generate all listed trees' positions, types, and attributes. Specifically, successful states
	- pass filter2, and
	- for each tree listed, have a state between -.maxCalls and .maxCalls advancements away from the tested state that could generate the tree's coordinates, type, and attributes.
   filter3_numberOfResultsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in filter3_numberOfResultsPerRun.*/
// TODO: Unify with filter7
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter3() {
	uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
#else
void *filter3(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= filter2_numberOfResultsPerRun) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t seed = filterInputs[index];

	uint32_t found = 0;
	Random random = Random::withSeed(seed);
	int32_t calls = POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls;
	if (calls & 256) random.skip<-256>();
	if (calls & 128) random.skip<-128>();
	if (calls & 64) random.skip<-64>();
	if (calls & 32) random.skip<-32>();
	if (calls & 16) random.skip<-16>();
	if (calls & 8) random.skip<-8>();
	if (calls & 4) random.skip<-4>();
	if (calls & 2) random.skip<-2>();
	if (calls & 1) random.skip<-1>();

	for (int32_t __currentCall = -POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; __currentCall <= POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; ++__currentCall) {
		#pragma unroll
		for (uint32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
			const TreeChunkPosition &tree = POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j];

			Random treeRandom(random);
			if (tree.testXZTypeAndAttributes(treeRandom, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
		}
		random.skip<1>();
	}

	if (found != ((UINT32_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) - 1)) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter3_numberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those that, for some combination of valid and invalid tree generation attempts, generate a tree chunk containing all trees in POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].
   treechunkFilter_numberOfResultsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in treechunkFilter_numberOfResultsPerRun.*/
// TODO: See if unifying with filter8 would be possible?
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void treechunkFilter() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t calls = index % (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls + 1);
	index /= (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls + 1);

	uint32_t validIngamePositionsMask = index % (UINT64_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount);
	if (getNumberOfOnesIn(validIngamePositionsMask) < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) return;
	index /= UINT64_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount;
	if (index >= filter3_numberOfResultsPerRun) return;
	uint64_t originalSeed = filterInputs[index];
#else
void *treechunkFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
	if (index >= filter3_numberOfResultsPerRun) return NULL;
	uint64_t originalSeed = filterInputs[index];
	for (uint32_t validIngamePositionsMask = 0; validIngamePositionsMask < UINT64_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount; ++validIngamePositionsMask) {
		if (getNumberOfOnesIn(validIngamePositionsMask) < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) continue;
		for (uint32_t calls = 0; calls <= POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; ++calls) {
#endif
			Random random = Random::withSeed(originalSeed);
			if (calls & 256) random.skip<-256>();
			if (calls & 128) random.skip<-128>();
			if (calls & 64) random.skip<-64>();
			if (calls & 32) random.skip<-32>();
			if (calls & 16) random.skip<-16>();
			if (calls & 8) random.skip<-8>();
			if (calls & 4) random.skip<-4>();
			if (calls & 2) random.skip<-2>();
			if (calls & 1) random.skip<-1>();
			uint64_t seed = random.seed;

			uint32_t treeCount = biomeTreeCount(random, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
			// If fewer trees are predicted to generate than there are in the input data, skip to the next seed.
			if (validIngamePositionsMask >> treeCount)
				#if (CUDA_IS_PRESENT)
					return;
				#else
					continue;
				#endif

			uint32_t found = 0;

			#pragma unroll
			for (uint32_t i = 0; i < treeCount; i++) {
				bool isValidIngamePosition = (validIngamePositionsMask >> i) & 1;

				if (isValidIngamePosition) {
					#pragma unroll
					for (int32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
						Random treeRandom(random);
						if (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZTypeAndAttributes(treeRandom, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT64_C(1) << j;
					}
				}

				TreeChunkPosition::skip(random, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, isValidIngamePosition, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
			}

			if (getNumberOfOnesIn(found) != POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions)
				#if (CUDA_IS_PRESENT)
					return;
				#else
					continue;
				#endif

			if (COLLAPSE_NEARBY_SEEDS_FLAG) {
				uint32_t seedMask = UINT32_C(1) << (index & 31);
				if (atomicOr(&filter3_masks[index / 32], seedMask) & seedMask)
				#if (CUDA_IS_PRESENT)
					return;
				#else
					continue;
				#endif
			}

			uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&treechunkFilter_numberOfResultsPerRun), 1);
			if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				#if CUDA_IS_PRESENT
					return;
				#else
					return NULL;
				#endif
			}
			filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
		}
	}
	return NULL;
#endif
}

/* Reverses population seeds (returned by Filter 4) back to structure seeds.
   totalStructureSeedsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in totalStructureSeedsPerRun.*/
// TODO: Maybe store results directly in stage2FilterInputs[] if more than one chunk is in the input data, removing the need for filter5?
// I'd need to ensure this filter can't return duplicate structure seeds, though, and I'd still need to save the structure seeds in filterInputs[] for filter8.
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void reversePopulationSeeds() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *reversePopulationSeeds(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= treechunkFilter_numberOfResultsPerRun) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t internalSeed = filterInputs[index];

	Random random = Random::withSeed(internalSeed);
	for (int32_t i = 0; i <= 10000*static_cast<int32_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version <= Version::v1_12_2); i++) {
		uint64_t populationSeed = random.seed ^ LCG::MULTIPLIER;
		if (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version <= Version::v1_12_2) reverse_1_12_populationSeed(populationSeed, static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkX), static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkZ));
		else {
			populationSeed = (populationSeed - POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].salt) & LCG::MASK;
			reverse_1_13_populationSeed(populationSeed, static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkX*16), static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkZ*16));
		}
		random.skip<-1>();
	}
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}


/*
	For each structure seed, get the corresponding population chunk seed.
	filter5_numberOfResultsPerRun must be set to 0 beforehand.
	Values are read from filterInputs[] and outputted to stage2filterInputs[], with the final count being stored in filter5_numberOfResultsPerRun.
*/
// TODO: This isn't necessary if I can ensure reversePopulationSeeds() doesn't return duplicate structure seeds (see reversePopulationSeeds() for caveats, though)
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter5() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter5(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= totalStructureSeedsPerRun) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t structureSeed = filterInputs[index];

	Random random;
	if (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version <= Version::v1_12_2) {
		random.setSeed(structureSeed);
		uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
		uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
		uint64_t populationSeed = static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkX) * a + static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkZ) * b ^ structureSeed;
		random.setSeed(populationSeed);
	} else {
		random.setSeed(structureSeed);
		uint64_t a = random.nextLong() | 1;
		uint64_t b = random.nextLong() | 1;
		uint64_t populationSeed = static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkX * 16) * a + static_cast<uint64_t>(POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkZ * 16) * b ^ structureSeed;
		uint64_t decorationSeed = populationSeed + POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].salt;
		random.setSeed(decorationSeed);
	}

	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter5_numberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	stage2filterInputs[resultIndex] = {index, random.seed};
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* ...
   filter6_numberOfResultsPerRun must be set to 0 beforehand.
   Values are read from stage2filterInputs[] and outputted to stage2filterResults[], with the final count being stored in filter6_numberOfResultsPerRun.*/
// TODO: Unify with filter2
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter6() {
	uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
#else
void *filter6(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	uint32_t skip = index % POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].rangeOfPossibleSkips;
	index /= POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].rangeOfPossibleSkips;

	if (index >= filter5_numberOfResultsPerRun) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t seed = stage2filterInputs[index].treechunkSeed;

	Random random = Random::withSeed(seed);
	if (skip & 8192) random.skip<8192>();
	if (skip & 4096) random.skip<4096>();
	if (skip & 2048) random.skip<2048>();
	if (skip & 1024) random.skip<1024>();
	if (skip &  512) random.skip<512>();
	if (skip &  256) random.skip<256>();
	if (skip &  128) random.skip<128>();
	if (skip &   64) random.skip<64>();
	if (skip &   32) random.skip<32>();
	if (skip &   16) random.skip<16>();
	if (skip &    8) random.skip<8>();
	if (skip &    4) random.skip<4>();
	if (skip &    2) random.skip<2>();
	if (skip &    1) random.skip<1>();
	seed = random.seed;

	uint32_t found = 0;
	for (int32_t i = 0; i <= POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; i++) {
		#pragma unroll
		for (uint32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
			Random treeRandom(random);
			if (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZAndType(treeRandom, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT64_C(1) << j;
		}
		random.skip<1>();
	}

	if (found != ((UINT32_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) - 1)) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter6_numberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	stage2filterResults[resultIndex] = {stage2filterInputs[index].structureSeedIndex, seed};
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* ...
   filter7_numberOfResultsPerRun must be set to 0 beforehand.
   Values are read from stage2filterInputs[] and outputted to stage2filterResults[], with the final count being stored in filter7_numberOfResultsPerRun.*/
// TODO: Unify with filter3
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter7() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter7(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= filter6_numberOfResultsPerRun) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t seed = stage2filterInputs[index].treechunkSeed;

	uint32_t found = 0;
	Random random = Random::withSeed(seed);
	for (int32_t i = 0; i <= POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; i++) {
		#pragma unroll
		for (uint32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
			Random treeRandom(random);
			if (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZTypeAndAttributes(treeRandom, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT64_C(1) << j;
		}
		random.skip<1>();
	}

	if (found != ((UINT32_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) - 1)) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter7_numberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
	stage2filterResults[resultIndex] = {stage2filterInputs[index].structureSeedIndex, seed};
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* ...
   totalStructureSeedsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] (structure seeds) and stage2filterInputs[] (treechunk seeds) and outputted to filterResults[], with the final count being stored in totalStructureSeedsPerRun.*/
// TODO: See if unifying with treechunkFilter would be possible?
// TODO: See if storing structure seeds directly in stage2FilterResults (instead of their indices) could be possible for filters 5, 6, and 7; that would remove the dependency on filterInputs[] 
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter8() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter8(void *dat) {
    uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

    uint32_t generated_mask = index % (UINT64_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount);
    index /= UINT64_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount;

    if (getNumberOfOnesIn(generated_mask) < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
    if (index >= filter7_numberOfResultsPerRun) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
    uint64_t seed = stage2filterInputs[index].treechunkSeed;

    Random random = Random::withSeed(seed);

    uint32_t treeCount = biomeTreeCount(random, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
    if (generated_mask >> treeCount) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
    uint32_t found = 0;

    #pragma unroll
    for (uint32_t i = 0; i < treeCount; i++) {
        bool generated = (generated_mask >> i) & 1;

        if (generated) {
            #pragma unroll
            for (int32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
                Random treeRandom(random);
                if (POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZTypeAndAttributes(treeRandom, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT64_C(1) << j;
            }
        }

        TreeChunkPosition::skip(random, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, generated, POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
    }
    if (getNumberOfOnesIn(found) != POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
    uint32_t results2_3_index = index;
    uint32_t results2_3_mask_mask = UINT32_C(1) << (results2_3_index & 31);
    if (atomicOr(&filter8_masks[results2_3_index / 32], results2_3_mask_mask) & results2_3_mask_mask) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
    uint32_t structureSeed = stage2filterInputs[index].structureSeedIndex;
    uint32_t structureSeedMask = UINT32_C(1) << (structureSeed & 31);
    if (atomicOr(&filter5_masks[structureSeed / 32], structureSeedMask) & structureSeedMask) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
    uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalStructureSeedsPerRun), 1);
    if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
		#if CUDA_IS_PRESENT
			return;
		#else
			return NULL;
		#endif
	}
    filterResults[resultIndex] = filterInputs[structureSeed];
#if (!CUDA_IS_PRESENT)
    return NULL;
#endif
}

/* Reverses structure seeds back to potential worldseeds.
   totalWorldseedsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in totalWorldseedsPerRun.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void biomeFilter() {
	uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	uint64_t topBits = index & 65535;
	index /= 65536;
#else
void *biomeFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif
	if (index >= totalStructureSeedsPerRun) return;
	Generator g;
	int biomes[8095]; // Largest possible output of Cubiomes' getMinCacheSize()
	Version lastVersion;
	uint64_t resultIndex;
#if (!CUDA_IS_PRESENT)
	for (uint64_t topBits = 0; topBits < 65536; ++topBits) {
#endif
		uint64_t seed = (topBits << 48) + filterInputs[index];
		for (uint32_t i = 0; i < POPULATION_CHUNKS_DATA.numberOfTreeChunks; ++i) {
			Version currentVersion = POPULATION_CHUNKS_DATA.treeChunks[i].version;
			if (!i
			#if (!CUDA_IS_PRESENT)
				&& !topBits
			#endif
			 || currentVersion != lastVersion) {
				setupGenerator(g, versionToCubiomesVersion(currentVersion), largeBiomesFlag);
				applySeed(g, seed);
				lastVersion = currentVersion;
			}
			printf("(Here?) ");
			int32_t populationChunkBlockX = getMinBlockCoordinate(POPULATION_CHUNKS_DATA.treeChunks[i].populationChunkX, currentVersion);
			int32_t populationChunkBlockZ = getMinBlockCoordinate(POPULATION_CHUNKS_DATA.treeChunks[i].populationChunkZ, currentVersion);
			Range populationChunkRange = {1, populationChunkBlockX, populationChunkBlockZ, 16, 16, 64, 1};
			if (genBiomes(g, biomes, populationChunkRange)) {
				#if CUDA_IS_PRESENT
					return;
				#else
					goto nextTopBits;
				#endif
			}

			// Ensure at least 1 block within the chunk is the desired biome
			int desiredBiome = static_cast<int>(POPULATION_CHUNKS_DATA.treeChunks[i].biome);
			bool containsBiome = false;
			for (uint32_t j = 0; j < 255; ++j) containsBiome |= (biomes[j] == desiredBiome);
			if (!containsBiome) {
				#if CUDA_IS_PRESENT
					return;
				#else
					goto nextTopBits;
				#endif
			}

			// For each tree, ensure its corresponding biome can generate dirt/grass to spawn a tree on
			#pragma unroll
			for (uint32_t j = 0; j < POPULATION_CHUNKS_DATA.treeChunks[i].numberOfTreePositions; ++j) {
				uint32_t correspondingIndex = 16*POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkZOffset + POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkXOffset;
				if (!biomeHasGrass(biomes[correspondingIndex])) {
					#if CUDA_IS_PRESENT
						return;
					#else
						goto nextTopBits;
					#endif
				}
			}
		}
		resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalWorldseedsPerRun), 1);
		if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
			#if CUDA_IS_PRESENT
				return;
			#else
				return NULL;
			#endif
		}
		filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
		nextTopBits:
	}
	return NULL;
#endif
}

#endif