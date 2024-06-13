#ifndef __FILTERS_CUH
#define __FILTERS_CUH

#include "Population Chunk Reversal Logic.cuh"
#include "Biome Logic.cuh"

// TODO: Add checks to ensure filters' inputs and outputs are distinct
#define PRINT_INPUT filterStorageA

__managed__ uint64_t filter1_numberOfResultsThisWorkerSet = 0;
uint64_t filter1_totalResultsThisRun = 0;
#define FILTER_1_OUTPUT filterStorageA

__managed__ uint64_t filter2_numberOfResultsThisWorkerSet = 0;
uint64_t filter2_totalResultsThisRun = 0;
#define FILTER_2_INPUT  filterStorageA
#define FILTER_2_OUTPUT filterStorageB

__managed__ uint64_t filter3_numberOfResultsThisWorkerSet = 0;
uint64_t filter3_totalResultsThisRun = 0;
#define FILTER_3_INPUT  filterStorageB
#define FILTER_3_OUTPUT filterStorageA

__device__ uint32_t filter3_masks[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN/32];
__managed__ uint64_t treechunkFilter_numberOfResultsThisWorkerSet = 0;
uint64_t treechunkFilter_totalResultsThisRun = 0;
#define TREECHUNK_FILTER_INPUT  filterStorageA
#define TREECHUNK_FILTER_OUTPUT filterStorageB

// TODO: Migrate Population Chunk Reversal settings to here (then move filter storage arrays from Settings_and_Input_Data_Processing.cuh)

__managed__ uint64_t filter5_numberOfResultsThisWorkerSet = 0;
uint64_t filter5_totalResultsThisRun = 0;
#define FILTER_5_INPUT  filterStorageB
#define FILTER_5_OUTPUT filterDoubleStorageA

__managed__ uint64_t filter6_numberOfResultsThisWorkerSet = 0;
uint64_t filter6_totalResultsThisRun = 0;
#define FILTER_6_INPUT  filterDoubleStorageA
#define FILTER_6_OUTPUT filterDoubleStorageB

__managed__ uint64_t filter7_numberOfResultsThisWorkerSet = 0;
uint64_t filter7_totalResultsThisRun = 0;
#define FILTER_7_INPUT  filterDoubleStorageB
#define FILTER_7_OUTPUT filterDoubleStorageA

__device__ uint32_t filter5_masks[(ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN + 31) / 32];
__device__ uint32_t filter8_masks[(ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN + 31) / 32];
#define FILTER_8_STRUCTURESEED_INPUT filterStorageB
#define FILTER_8_TREECHUNK_INPUT filterDoubleStorageA
#define FILTER_8_OUTPUT filterStorageA

#define WORLDSEED_FILTER_INPUT  filterStorageA
#define WORLDSEED_FILTER_OUTPUT filterStorageB

/* Filters possible internal Random states for those that could generate the first tree in ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex]. Specifically, successful states
	- would choose the first tree's x-coordinate (if RELATIVE_COORDINATES_MODE is false);
	- on the next call, would choose the first tree's z-coordinate (if RELATIVE_COORDINATES_MODE is false);
	- on the next call(s), would choose the first tree's type; and
	- on subsequent calls, would choose all of the first tree's attributes.
   filter1_numberOfResultsThisWorkerSet must be set to 0 beforehand.
   Values are generated internally and outputted to FILTER_1_OUTPUT[], with the final count being stored in filter1_numberOfResultsThisWorkerSet.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter1(const uint64_t start) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t seed = (RELATIVE_COORDINATES_MODE ? UINT64_C(0) : (static_cast<uint64_t>(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[0].populationChunkXOffset) << 44)) + start + index;
	// From Cortex's TreeCracker, but is both slower and doesn't seem to work at the moment...
	// uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x + start;
	// int64_t latticeX = (index % PARALLELOGRAM_SIZE.x) + PARALLELOGRAM_LOWEST_CORNER.x;
	// int64_t latticeZ = (index/PARALLELOGRAM_SIZE.x) + PARALLELOGRAM_LOWEST_CORNER.z;
	// if (VECTOR_1.x * latticeZ < VECTOR_1.z * latticeX) latticeZ += PARALLELOGRAM_SIZE.z;
	// if (VECTOR_2.x * latticeZ < VECTOR_2.z * latticeX) {
	// 	latticeX += VECTOR_1.x;
	// 	latticeZ += VECTOR_1.z;
	// }
	// latticeX += ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[0].treePositions[0].populationChunkXOffset * VECTOR_1.x + ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[0].treePositions[0].populationChunkZOffset * VECTOR_2.x;
	// latticeZ += ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[0].treePositions[0].populationChunkXOffset * VECTOR_1.z + ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[0].treePositions[0].populationChunkZOffset * VECTOR_2.z;

	// uint64_t seed = (latticeX * 7847617 + latticeZ * -18218081) & LCG::MASK;

#else
void *filter1(void *start) {
	ThreadData *star = static_cast<ThreadData *>(start);
	uint64_t seed = (RELATIVE_COORDINATES_MODE ? UINT64_C(0) : (static_cast<uint64_t>(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[0].populationChunkXOffset) << 44)) + star->start + star->index;
#endif

	Random treeRandom = Random::withSeed(seed);
	if (RELATIVE_COORDINATES_MODE) {
		if (!ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[0].testTypeAndAttributes(treeRandom, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) FILTER_RETURN;
		seed = Random::withSeed(seed).skip<-2>().seed; // Step back two calls to account for x call (nextInt(16)) and z call (nextInt(16))
	} else { // Absolute coordinates mode
		if (!ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[0].testZTypeAndAttributes(treeRandom, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) FILTER_RETURN;
	}
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter1_numberOfResultsThisWorkerSet), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
	FILTER_1_OUTPUT[resultIndex] = seed;
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
   filter2_numberOfResultsThisWorkerSet must be set to 0 beforehand.
   Values are read from FILTER_2_INPUT[] and outputted to FILTER_2_OUTPUT[], with the final count being stored in filter2_numberOfResultsThisWorkerSet.*/
// TODO: Unify with filter6
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter2() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter2(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= filter1_numberOfResultsThisWorkerSet) FILTER_RETURN;

	uint64_t seed = FILTER_2_INPUT[index];

	uint32_t found = 0;
	// Random random = Random::withSeed(seed);
	// int32_t calls = ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls;
	// if (calls & 256) random.skip<-256>();
	// if (calls & 128) random.skip<-128>();
	// if (calls & 64) random.skip<-64>();
	// if (calls & 32) random.skip<-32>();
	// if (calls & 16) random.skip<-16>();
	// if (calls & 8) random.skip<-8>();
	// if (calls & 4) random.skip<-4>();
	// if (calls & 2) random.skip<-2>();
	// if (calls & 1) random.skip<-1>();
	Random random = Random::withSeed(seed).skip(-ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls);

	for (int32_t _currentCall = -ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; _currentCall <= ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; ++_currentCall) {
		#pragma unroll
		for (uint32_t j = 0; j < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
			Random treeRandom(random);
			// if (RELATIVE_COORDINATES_MODE) {
				// treeRandom.skip<2>();
				// if (RELATIVE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testType(treeRandom, RELATIVE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, RELATIVE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
			// } else {
				if (ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZAndType(treeRandom, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
			// }
		}
		random.skip<1>();
	}

	if (found != ((UINT32_C(1) << ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) - 1)) FILTER_RETURN;
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter2_numberOfResultsThisWorkerSet), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
	FILTER_2_OUTPUT[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those whose surrounding states, within the maximum possible range of calls, could generate all listed trees' positions, types, and attributes. Specifically, successful states
	- pass filter2, and
	- for each tree listed, have a state between -.maxCalls and .maxCalls advancements away from the tested state that could generate the tree's coordinates, type, and attributes.
   filter3_numberOfResultsThisWorkerSet must be set to 0 beforehand.
   Values are read from FILTER_3_INPUT[] and outputted to FILTER_3_OUTPUT[], with the final count being stored in filter3_numberOfResultsThisWorkerSet.*/
// TODO: Unify with filter7
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter3() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter3(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= filter2_numberOfResultsThisWorkerSet) FILTER_RETURN;
	uint64_t seed = FILTER_3_INPUT[index];

	uint32_t found = 0;
	// Random random = Random::withSeed(seed);
	// int32_t calls = ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls;
	// if (calls & 256) random.skip<-256>();
	// if (calls & 128) random.skip<-128>();
	// if (calls & 64) random.skip<-64>();
	// if (calls & 32) random.skip<-32>();
	// if (calls & 16) random.skip<-16>();
	// if (calls & 8) random.skip<-8>();
	// if (calls & 4) random.skip<-4>();
	// if (calls & 2) random.skip<-2>();
	// if (calls & 1) random.skip<-1>();
	Random random = Random::withSeed(seed).skip(-ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls);

	for (int32_t __currentCall = -ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; __currentCall <= ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; ++__currentCall) {
		#pragma unroll
		for (uint32_t j = 0; j < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
			const TreeChunkPosition &tree = ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j];

			Random treeRandom(random);
			if (tree.testXZTypeAndAttributes(treeRandom, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
		}
		random.skip<1>();
	}

	if (found != ((UINT32_C(1) << ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) - 1)) FILTER_RETURN;
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter3_numberOfResultsThisWorkerSet), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
	FILTER_3_OUTPUT[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those that, for some combination of valid and invalid tree generation attempts, generate a tree chunk containing all trees in ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].
   treechunkFilter_numberOfResultsThisWorkerSet must be set to 0 beforehand.
   Values are read from TREECHUNK_FILTER_INPUT[] and outputted to TREECHUNK_FILTER_OUTPUT[], with the final count being stored in treechunkFilter_numberOfResultsThisWorkerSet.*/
// TODO: See if unifying with filter8 would be possible?
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void treechunkFilter() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t calls = index % (ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls + 1);
	index /= (ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls + 1);

	uint32_t validIngamePositionsMask = getLowestBitsOf(index, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount);
	if (getNumberOfOnesIn(validIngamePositionsMask) < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) FILTER_RETURN;
	index /= twoToThePowerOf(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount);
	if (index >= filter3_numberOfResultsThisWorkerSet) FILTER_RETURN;
	uint64_t originalSeed = TREECHUNK_FILTER_INPUT[index];
#else
void *treechunkFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
	if (index >= filter3_numberOfResultsThisWorkerSet) FILTER_RETURN;
	uint64_t originalSeed = TREECHUNK_FILTER_INPUT[index];
	for (uint32_t validIngamePositionsMask = 0; validIngamePositionsMask < twoToThePowerOf(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount); ++validIngamePositionsMask) {
		if (getNumberOfOnesIn(validIngamePositionsMask) < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) continue;
		for (uint32_t calls = 0; calls <= ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; ++calls) {
#endif
			Random random = Random::withSeed(originalSeed);
			if (calls & 256) random.skip<-256>();
			if (calls & 128) random.skip<-128>();
			if (calls &  64) random.skip<- 64>();
			if (calls &  32) random.skip<- 32>();
			if (calls &  16) random.skip<- 16>();
			if (calls &   8) random.skip<-  8>();
			if (calls &   4) random.skip<-  4>();
			if (calls &   2) random.skip<-  2>();
			if (calls &   1) random.skip<-  1>();
			// TODO: Not sure why this wouldn't work yet
			// Random random = Random::withSeed(originalSeed).skip(-calls);
			uint64_t seed = random.seed;

			uint32_t treeCount = biomeTreeCount(random, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
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
					for (int32_t j = 0; j < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
						Random treeRandom(random);
						if (ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZTypeAndAttributes(treeRandom, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
					}
				}

				TreeChunkPosition::skip(random, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, isValidIngamePosition, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
			}

			if (getNumberOfOnesIn(found) != ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions)
				#if (CUDA_IS_PRESENT)
					return;
				#else
					continue;
				#endif

			if (ABSOLUTE_POPULATION_CHUNKS_DATA.collapseNearbySeedsFlag) {
				uint32_t seedMask = UINT32_C(1) << (index & 31);
				if (atomicOr(&filter3_masks[index / 32], seedMask) & seedMask)
				#if (CUDA_IS_PRESENT)
					return;
				#else
					continue;
				#endif
			}

			uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&treechunkFilter_numberOfResultsThisWorkerSet), 1);
			if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
			TREECHUNK_FILTER_OUTPUT[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
		}
	}
	return NULL;
#endif
}

/* Reverses population seeds (returned by Filter 4) back to structure seeds.
   totalStructureSeedsThisWorkerSet must be set to 0 beforehand.
   Values are read from POUPLATION_REVERSAL_INPUT[] and outputted to POPULATION_REVERSAL_OUTPUT[], with the final count being stored in totalStructureSeedsThisWorkerSet.*/
// TODO: Maybe store results directly in stage2FilterInputs[] if more than one chunk is in the input data, removing the need for filter5?
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void reversePopulationSeeds() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *reversePopulationSeeds(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= treechunkFilter_numberOfResultsThisWorkerSet) FILTER_RETURN;
	uint64_t internalSeed = POPULATION_REVERSAL_INPUT[index];

	Random random = Random::withSeed(internalSeed);
	stepBackMinPopulationCalls(random, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
	for (int32_t i = 0; i < getPopulationCallsRange(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version); ++i) {
		uint64_t populationSeed = ((random.seed ^ LCG::MULTIPLIER) - ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].salt) & LCG::MASK;
		reversePopulationSeed(populationSeed, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkX*(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version <= static_cast<Version>(ExperimentalVersion::v1_12_2) ? 1 : 16), ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkZ*(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version <= static_cast<Version>(ExperimentalVersion::v1_12_2) ? 1 : 16), ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
		random.skip<-1>();
	}
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}


/*
	For each structure seed, get the corresponding population chunk seed.
	filter5_numberOfResultsThisWorkerSet must be set to 0 beforehand.
	Values are read from FILTER_5_INPUT[] and outputted to FILTER_5_OUTPUT[], with the final count being stored in filter5_numberOfResultsThisWorkerSet.
*/
// TODO: This isn't necessary if I can ensure reversePopulationSeeds() doesn't return duplicate structure seeds (see reversePopulationSeeds() for caveats, though)
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter5() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter5(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= totalStructureSeedsThisWorkerSet) FILTER_RETURN;
	uint64_t structureSeed = FILTER_5_INPUT[index];

	Random random(structureSeed);
	if (ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version <= static_cast<Version>(ExperimentalVersion::v1_12_2)) {
		uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
		uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
		uint64_t populationSeed = static_cast<uint64_t>(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkX) * a + static_cast<uint64_t>(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkZ) * b ^ structureSeed;
		random.setSeed(populationSeed);
	} else {
		uint64_t a = random.nextLong() | 1;
		uint64_t b = random.nextLong() | 1;
		uint64_t populationSeed = static_cast<uint64_t>(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkX * 16) * a + static_cast<uint64_t>(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].populationChunkZ * 16) * b ^ structureSeed;
		uint64_t decorationSeed = populationSeed + ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].salt;
		random.setSeed(decorationSeed);
	}

	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter5_numberOfResultsThisWorkerSet), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
	// FILTER_5_OUTPUT[resultIndex] = (sizeof(*FILTER_5_OUTPUT) > sizeof(uint64_t)) ? {index, random.seed} : random.seed;
	FILTER_5_OUTPUT[resultIndex] = {index, random.seed};
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* ...
   filter6_numberOfResultsThisWorkerSet must be set to 0 beforehand.
   Values are read from FILTER_6_INPUT[] and outputted to FILTER_6_OUTPUT[], with the final count being stored in filter6_numberOfResultsThisWorkerSet.*/
// TODO: Unify with filter2
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter6() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter6(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	uint32_t skip = index % ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].rangeOfPossibleSkips;
	index /= ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].rangeOfPossibleSkips;

	if (index >= filter5_numberOfResultsThisWorkerSet) FILTER_RETURN;
	uint64_t seed = FILTER_6_INPUT[index].treechunkSeed;

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
	for (int32_t i = 0; i <= ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; i++) {
		#pragma unroll
		for (uint32_t j = 0; j < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
			Random treeRandom(random);
			if (ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZAndType(treeRandom, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
		}
		random.skip<1>();
	}

	if (found != ((UINT32_C(1) << ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) - 1)) FILTER_RETURN;
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter6_numberOfResultsThisWorkerSet), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
	FILTER_6_OUTPUT[resultIndex] = {FILTER_6_INPUT[index].structureSeedIndex, seed};
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* ...
   filter7_numberOfResultsThisWorkerSet must be set to 0 beforehand.
   Values are read from FILTER_7_INPUT[] and outputted to FILTER_7_OUTPUT[], with the final count being stored in filter7_numberOfResultsThisWorkerSet.*/
// TODO: Unify with filter3
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter7() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter7(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= filter6_numberOfResultsThisWorkerSet) FILTER_RETURN;
	uint64_t seed = FILTER_7_INPUT[index].treechunkSeed;

	uint32_t found = 0;
	Random random = Random::withSeed(seed);
	for (int32_t i = 0; i <= ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxCalls; i++) {
		#pragma unroll
		for (uint32_t j = 0; j < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
			Random treeRandom(random);
			if (ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZTypeAndAttributes(treeRandom, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
		}
		random.skip<1>();
	}

	if (found != ((UINT32_C(1) << ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) - 1)) FILTER_RETURN;
	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter7_numberOfResultsThisWorkerSet), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
	FILTER_7_OUTPUT[resultIndex] = {FILTER_7_INPUT[index].structureSeedIndex, seed};
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* ...
   totalStructureSeedsThisWorkerSet must be set to 0 beforehand.
   Values are read from FILTER_8_STRUCTURESEED_INPUT[] (structure seeds) and FILTER_8_TREECHUNK_INPUT[] (treechunk seeds) and outputted to FILTER_8_OUTPUT[], with the final count being stored in totalStructureSeedsThisWorkerSet.*/
// TODO: See if unifying with treechunkFilter would be possible?
// TODO: See if storing structure seeds directly in FILTER8_TREECHUNK_INPUT (instead of their indices) could be possible for filters 5, 6, and 7; that would remove the dependency on FILTER8_STRUCTURESEED_INPUT[] 
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter8() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter8(void *dat) {
    uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

    uint32_t generated_mask = getLowestBitsOf(index, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount);
    index /= twoToThePowerOf(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount);

    if (getNumberOfOnesIn(generated_mask) < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions || index >= filter7_numberOfResultsThisWorkerSet) FILTER_RETURN;
    uint64_t seed = FILTER_8_TREECHUNK_INPUT[index].treechunkSeed;

    Random random = Random::withSeed(seed);

    uint32_t treeCount = biomeTreeCount(random, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
    if (generated_mask >> treeCount) FILTER_RETURN;
    uint32_t found = 0;

    #pragma unroll
    for (uint32_t i = 0; i < treeCount; i++) {
        bool generated = (generated_mask >> i) & 1;

        if (generated) {
            #pragma unroll
            for (int32_t j = 0; j < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions; j++) {
                Random treeRandom(random);
                if (ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].treePositions[j].testXZTypeAndAttributes(treeRandom, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version)) found |= UINT32_C(1) << j;
            }
        }

        TreeChunkPosition::skip(random, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].biome, generated, ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].version);
    }
    if (getNumberOfOnesIn(found) != ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].numberOfTreePositions) FILTER_RETURN;
    uint32_t results2_3_index = index;
    uint32_t results2_3_mask_mask = UINT32_C(1) << (results2_3_index & 31);
    if (atomicOr(&filter8_masks[results2_3_index / 32], results2_3_mask_mask) & results2_3_mask_mask) FILTER_RETURN;
    uint32_t structureSeed = FILTER_8_TREECHUNK_INPUT[index].structureSeedIndex;
    uint32_t structureSeedMask = UINT32_C(1) << (structureSeed & 31);
    if (atomicOr(&filter5_masks[structureSeed / 32], structureSeedMask) & structureSeedMask) FILTER_RETURN;
    uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalStructureSeedsThisWorkerSet), 1);
    if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
    FILTER_8_OUTPUT[resultIndex] = FILTER_8_STRUCTURESEED_INPUT[structureSeed];
#if (!CUDA_IS_PRESENT)
    return NULL;
#endif
}

/* Reverses structure seeds back to potential worldseeds.
   totalWorldseedsThisWorkerSet must be set to 0 beforehand.
   Values are read from WORLDSEED_FILTER_INPUT[] and outputted to WORLDSEED_FILTER_OUTPUT[], with the final count being stored in totalWorldseedsThisWorkerSet.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void biomeFilter() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *biomeFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif
	uint64_t topBits = 0;
	if (static_cast<int32_t>(TYPES_TO_OUTPUT) & (static_cast<int32_t>(ExperimentalOutputType::Random_Worldseeds) | static_cast<int32_t>(ExperimentalOutputType::All_Worldseeds))) {
		topBits = index & 65535;
		index /= 65536;
	}
	if (index >= totalStructureSeedsThisWorkerSet) FILTER_RETURN;
	Generator g;
	int biomes[8095]; // Largest possible output of Cubiomes' getMinCacheSize()
	Version lastVersion = static_cast<Version>(-1); // Guaranteed invalid
	uint64_t resultIndex;
#if (!CUDA_IS_PRESENT)
	for (uint64_t topBits = 0; topBits < 65536; ++topBits) {
#endif
		uint64_t seed = (topBits << 48) + WORLDSEED_FILTER_INPUT[index];
		for (uint32_t i = 0; i < ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks; ++i) {
			Version currentVersion = ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[i].version;
			if (!i
			#if (!CUDA_IS_PRESENT)
				&& !topBits
			#endif
			 || currentVersion != lastVersion) {
				// printf("A ");
				setupGenerator(g, versionToCubiomesVersion(currentVersion), largeBiomesFlag);
				// printf("B ");
				applySeed(g, seed);
				// printf("C ");
				lastVersion = currentVersion;
			}
			// printf("(Here?) ");
			int32_t populationChunkBlockX = getMinBlockCoordinate(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[i].populationChunkX, currentVersion);
			int32_t populationChunkBlockZ = getMinBlockCoordinate(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[i].populationChunkZ, currentVersion);
			Range populationChunkRange = {1, populationChunkBlockX, populationChunkBlockZ, 16, 16, 64, 1};
			if (genBiomes(g, biomes, populationChunkRange)) {
				#if CUDA_IS_PRESENT
					return;
				#else
					goto nextTopBits;
				#endif
			}

			// Ensure at least 1 block within the chunk is the desired biome
			int desiredBiome = static_cast<int>(ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[i].biome);
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
			for (uint32_t j = 0; j < ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[i].numberOfTreePositions; ++j) {
				uint32_t correspondingIndex = 16*ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkZOffset + ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[i].treePositions[j].populationChunkXOffset;
				if (!biomeHasDirtOrGrass(biomes[correspondingIndex])) {
					#if CUDA_IS_PRESENT
						return;
					#else
						goto nextTopBits;
					#endif
				}
			}
		}
		resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&totalWorldseedsThisWorkerSet), 1);
		if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) FILTER_RETURN;
		WORLDSEED_FILTER_OUTPUT[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
		nextTopBits:
	}
	return NULL;
#endif
}

// Prints the number of hits each filter had over a single partial run. Also resets each counter for the next partial run.
void printFilterResultCounts() {
	if (SILENT_MODE) return;
	// Filters 1-4 are used regardless of one's settings
	std::fprintf(stderr, "(Single Tree Filter: %" PRIu64 " result%s", filter1_totalResultsThisRun, getPlural(filter1_totalResultsThisRun));
	if (!filter1_totalResultsThisRun) goto end;
	filter1_totalResultsThisRun = 0;

	std::fprintf(stderr, ", \tCoords/Type Filter: %" PRIu64 " result%s", filter2_totalResultsThisRun, getPlural(filter2_totalResultsThisRun));
	if (!filter2_totalResultsThisRun) goto end;
	filter2_totalResultsThisRun = 0;

	std::fprintf(stderr, ", \tCoords/Type/Attributes Filter: %" PRIu64 " result%s", filter3_totalResultsThisRun, getPlural(filter3_totalResultsThisRun));
	if (!filter3_totalResultsThisRun) goto end;
	filter3_totalResultsThisRun = 0;

	std::fprintf(stderr, ", \tTreechunk Filter: %" PRIu64 " result%s", treechunkFilter_totalResultsThisRun, getPlural(treechunkFilter_totalResultsThisRun));
	if (!treechunkFilter_totalResultsThisRun) goto end;
	treechunkFilter_totalResultsThisRun = 0;
	if (static_cast<int32_t>(ACTUAL_TYPES_TO_OUTPUT) <= 2*static_cast<int32_t>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds) - 1) goto end;

	// If more than just treechunk seeds are requested, population chunk reversal will also be used
	std::fprintf(stderr, ", \tPopulation Chunk Reversal: %" PRIu64 " result%s", ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? totalStructureSeedsThisRun : filter5_totalResultsThisRun, getPlural(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? totalStructureSeedsThisRun : filter5_totalResultsThisRun));
	if (!filter5_totalResultsThisRun) goto end;
	// If the input data has multiple treechunks, filters 5-8 will also be used
	if (1 < ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks) {
		std::fprintf(stderr, "\n Treechunk Recovery: %" PRIu64 " result%s", filter5_totalResultsThisRun, getPlural(filter5_totalResultsThisRun));
		filter5_totalResultsThisRun = 0;

		std::fprintf(stderr, ", \tCoords/Type Filter: %" PRIu64 " result%s", filter6_totalResultsThisRun, getPlural(filter6_totalResultsThisRun));
		if (!filter6_totalResultsThisRun) goto end;
		filter6_totalResultsThisRun = 0;

		std::fprintf(stderr, ", \tCoords/Type/Attributes Filter: %" PRIu64 " result%s", filter7_totalResultsThisRun, getPlural(filter7_totalResultsThisRun));
		if (!filter7_totalResultsThisRun) goto end;
		filter7_totalResultsThisRun = 0;

		std::fprintf(stderr, ", \tTreechunk Filter: %" PRIu64 " result%s", totalStructureSeedsThisRun, getPlural(totalStructureSeedsThisRun));
		if (!totalStructureSeedsThisRun) goto end;
	}
	totalStructureSeedsThisRun = 0;
	if (static_cast<int32_t>(ACTUAL_TYPES_TO_OUTPUT) <= 2*static_cast<int32_t>(OutputType::Structure_Seeds) - 1) goto end;

	// If more than just treechunk and/or structure seeds are requested, biome filter will also be used
	std::fprintf(stderr, ", \tBiome Filter: %" PRIu64 " result%s", totalWorldseedsThisRun, getPlural(totalWorldseedsThisRun));
	totalWorldseedsThisRun = 0;

	end: std::fprintf(stderr, ")\n\n");
}

#endif