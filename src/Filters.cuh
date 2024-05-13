#ifndef __FILTERS_CUH
#define __FILTERS_CUH

#include "GenerateAndValidateData.cuh"
#include "ChunkRandomReversalFilter.cuh"
#include "BiomesFilter.cuh"

__device__ uint32_t theoreticalAttributesFilterMasks[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN / 32];

__managed__ uint64_t oneTreeFilterNumberOfResultsPerRun = 0;
__managed__ uint64_t theoreticalCoordsAndTypesFilterNumberOfResultsPerRun = 0;
__managed__ uint64_t theoreticalAttributesFilterNumberOfResultsPerRun = 0;
__managed__ uint64_t allAttributesFilterNumberOfResultsPerRun = 0;

struct Stage2Results {
    uint64_t structureSeedIndex, treeSeed;
};

__device__ Stage2Results filterInput2[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN]; // To prevent new results from accidentally erasing the old inputs mid-filter
__device__ Stage2Results filterResults2[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN]; // To prevent new results from accidentally erasing the old inputs mid-filter
__managed__ uint64_t filter_5_numberOfResultsPerRun = 0;
__managed__ uint64_t filter_6_numberOfResultsPerRun = 0;
__managed__ uint64_t filter_7_numberOfResultsPerRun = 0;
__managed__ uint64_t filter_8_numberOfResultsPerRun = 0;
__device__ uint32_t filter_5_masks[(ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN + 31) / 32];
__device__ uint32_t filter_8_masks[(ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN + 31) / 32];

#if (!CUDA_IS_PRESENT)
struct ThreadData {
	uint64_t index, start;
};
#endif

/* Filters possible internal Random states for those that could generate the first tree in FIRST_POPULATION_CHUNK. Specifically, successful states
    - would choose the first tree's x-coordinate;
    - on the next call, would choose the first tree's z-coordinate;
    - on the next call(s), would choose the first tree's type; and
    - on subsequent calls, would choose all of the first tree's attributes.
   oneTreeFilterNumberOfResultsPerRun must be set to 0 beforehand.
   Values are generated internally and outputted to filterResults[], with the final count being stored in oneTreeFilterNumberOfResultsPerRun.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void OneTreeFilter(const uint64_t start) {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t seed = (static_cast<uint64_t>(FIRST_POPULATION_CHUNK.treePositions[0].populationChunkXOffset) << 44) + start + index;
#else
void *OneTreeFilter(void *start) {
	ThreadData *star = static_cast<ThreadData *>(start);
	uint64_t seed = (static_cast<uint64_t>(FIRST_POPULATION_CHUNK.treePositions[0].populationChunkXOffset) << 44) + star->start + star->index;
#endif

	Random treeRandom = Random::withSeed(seed);
	if (!FIRST_POPULATION_CHUNK.treePositions[0].testZTypeAndAttributes(treeRandom, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version)) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;

	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&oneTreeFilterNumberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
	filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those that could generate the first tree in FIRST_POPULATION_CHUNK and all other trees' positions and types. Specifically, successful states
    - pass OneTreeFilter, and
    - for each other tree in the chunk, have a state between -MAX_CALLS and MAX_CALLS advancements away from the tested state that could generate the tree's
        - x-coordinate,
        - z-coordinate, and
        - type.
   theoreticalCoordsAndTypesFilterNumberOfResultsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in theoreticalCoordsAndTypesFilterNumberOfResultsPerRun.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void theoreticalCoordsAndTypesFilter() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *theoreticalCoordsAndTypesFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= oneTreeFilterNumberOfResultsPerRun) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
	uint64_t seed = filterInputs[index];

	uint32_t found = 0;
	Random random = Random::withSeed(seed).skip<-MAX_CALLS>();
	for (int32_t i = -MAX_CALLS; i <= MAX_CALLS; i++) {
		#pragma unroll
		for (uint32_t j = 0; j < FIRST_POPULATION_CHUNK.numberOfTreePositions; j++) {
			const TreeChunkPosition &tree = FIRST_POPULATION_CHUNK.treePositions[j];

			Random treeRandom(random);
			if (tree.testXZAndType(treeRandom, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version)) found |= UINT64_C(1) << j;
		}
		random.skip<1>();
	}

	if (getNumberOfOnesIn(found) != FIRST_POPULATION_CHUNK.numberOfTreePositions) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;

	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&theoreticalCoordsAndTypesFilterNumberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
	filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those that have the potential to generate each tree contained within FIRST_POPULATION_CHUNK. Specifically, successful states
    - pass theoreticalCoordsAndTypesFilter, and
    - for each other tree in the chunk, have a state between -MAX_CALLS and MAX_CALLS advancements away from the tested state that could generate the tree's attributes.
   theoreticalAttributesFilterNumberOfResultsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in theoreticalAttributesFilterNumberOfResultsPerRun.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void theoreticalAttributesFilter() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *theoreticalAttributesFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= theoreticalCoordsAndTypesFilterNumberOfResultsPerRun) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
	uint64_t seed = filterInputs[index];

	uint32_t found = 0;
	Random random = Random::withSeed(seed).skip<-MAX_CALLS>();
	for (int32_t i = -MAX_CALLS; i <= MAX_CALLS; i++) {
		#pragma unroll
		for (uint32_t j = 0; j < FIRST_POPULATION_CHUNK.numberOfTreePositions; j++) {
			const TreeChunkPosition &tree = FIRST_POPULATION_CHUNK.treePositions[j];

			Random treeRandom(random);
			if (tree.testXZTypeAndAttributes(treeRandom, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version)) found |= UINT64_C(1) << j;
		}
		random.skip<1>();
	}

	if (getNumberOfOnesIn(found) != FIRST_POPULATION_CHUNK.numberOfTreePositions) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;

	uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&theoreticalAttributesFilterNumberOfResultsPerRun), 1);
	if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
	filterResults[resultIndex] = seed;
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}

/* Filters possible internal Random states for those that generate a tree chunk matching FIRST_POPULATION_CHUNK.(?)
   allAttributesFilterNumberOfResultsPerRun must be set to 0 beforehand.
   Values are read from filterInputs[] and outputted to filterResults[], with the final count being stored in allAttributesFilterNumberOfResultsPerRun.*/
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void allAttributesFilter() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t calls = index % (MAX_CALLS + 1);
	index /= (MAX_CALLS + 1);

	uint32_t generatedMask = index % (UINT64_C(1) << MAX_TREE_COUNT);
	if (getNumberOfOnesIn(generatedMask) < FIRST_POPULATION_CHUNK.numberOfTreePositions) return;
	index /= UINT64_C(1) << MAX_TREE_COUNT;
	if (index >= theoreticalAttributesFilterNumberOfResultsPerRun) return;
	uint64_t originalSeed = filterInputs[index];
#else
void *allAttributesFilter(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
	if (index >= theoreticalAttributesFilterNumberOfResultsPerRun) return NULL;
	uint64_t originalSeed = filterInputs[index];
	for (uint32_t generatedMask = 0; generatedMask < UINT64_C(1) << MAX_TREE_COUNT; ++generatedMask) {
		if (getNumberOfOnesIn(generatedMask) < FIRST_POPULATION_CHUNK.numberOfTreePositions) continue;
		for (uint32_t calls = 0; calls <= MAX_CALLS; ++calls) {
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

			uint32_t treeCount = biomeTreeCount(random, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version);
			if (generatedMask >> treeCount)
				#if (CUDA_IS_PRESENT)
					return;
				#else
					continue;
				#endif

			uint32_t found = 0;

			#pragma unroll
			for (int32_t i = 0; i < treeCount; i++) {
				bool generated = (generatedMask >> i) & 1;

				if (generated) {
					#pragma unroll
					for (int32_t j = 0; j < FIRST_POPULATION_CHUNK.numberOfTreePositions; j++) {
						const TreeChunkPosition &tree = FIRST_POPULATION_CHUNK.treePositions[j];
						Random treeRandom(random);
						if (tree.testXZTypeAndAttributes(treeRandom, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version)) found |= UINT64_C(1) << j;
					}
				}

				TreeChunkPosition::skip(random, FIRST_POPULATION_CHUNK.biome, generated, FIRST_POPULATION_CHUNK.version);
			}

			if (getNumberOfOnesIn(found) != FIRST_POPULATION_CHUNK.numberOfTreePositions)
				#if (CUDA_IS_PRESENT)
					return;
				#else
					continue;
				#endif

			if (COLLAPSE_NEARBY_SEEDS_FLAG) {
				uint32_t seedMask = UINT32_C(1) << (index & 31);
				if (atomicOr(&theoreticalAttributesFilterMasks[index / 32], seedMask) & seedMask)
				#if (CUDA_IS_PRESENT)
					return;
				#else
					continue;
				#endif
			}

			uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&allAttributesFilterNumberOfResultsPerRun), 1);
			if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
				#if (!CUDA_IS_PRESENT)
					NULL
				#endif
				;
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
#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void reversePopulationSeeds() {
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *reversePopulationSeeds(void *dat) {
	uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

	if (index >= allAttributesFilterNumberOfResultsPerRun) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
	uint64_t internalSeed = filterInputs[index];

	Random random = Random::withSeed(internalSeed);
	for (int32_t i = 0; i <= 10000*static_cast<int32_t>(FIRST_POPULATION_CHUNK.version <= Version::v1_12_2); i++) {
		uint64_t populationSeed = random.seed ^ LCG::MULTIPLIER;
		if (FIRST_POPULATION_CHUNK.version <= Version::v1_12_2) reverse_1_12_populationSeed(populationSeed, static_cast<uint64_t>(FIRST_POPULATION_CHUNK.populationChunkX), static_cast<uint64_t>(FIRST_POPULATION_CHUNK.populationChunkZ));
		else {
			populationSeed = (populationSeed - SALT) & LCG::MASK;
			reverse_1_13_populationSeed(populationSeed, static_cast<uint64_t>(FIRST_POPULATION_CHUNK.populationChunkX*16), static_cast<uint64_t>(FIRST_POPULATION_CHUNK.populationChunkZ*16));
		}
		random.skip<-1>();
	}
#if (!CUDA_IS_PRESENT)
	return NULL;
#endif
}


#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter_5() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter_5(void *dat) {
    uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

    if (index >= totalStructureSeedsPerRun) return
		#if (!CUDA_IS_PRESENT)
			NULL
		#endif
		;
    uint64_t structureSeed = filterInputs[index];

    Random random;
    if (FIRST_POPULATION_CHUNK.version <= Version::v1_12_2) {
        random.setSeed(structureSeed);
        uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
        uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
        uint64_t populationSeed = static_cast<uint64_t>(FIRST_POPULATION_CHUNK.populationChunkX) * a + static_cast<uint64_t>(FIRST_POPULATION_CHUNK.populationChunkZ) * b ^ structureSeed;
        random.setSeed(populationSeed);
    } else {
        random.setSeed(structureSeed);
        uint64_t a = random.nextLong() | 1;
        uint64_t b = random.nextLong() | 1;
        uint64_t populationSeed = static_cast<uint64_t>(FIRST_POPULATION_CHUNK.populationChunkX * 16) * a + static_cast<uint64_t>(FIRST_POPULATION_CHUNK.populationChunkZ * 16) * b ^ structureSeed;
        uint64_t decorationSeed = populationSeed + SALT;
        random.setSeed(decorationSeed);
    }

    uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter_5_numberOfResultsPerRun), 1);
    if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
    filterResults2[resultIndex] = Stage2Results{index, random.seed};
#if (!CUDA_IS_PRESENT)
    return NULL;
#endif
}

#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter_6() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter_6(void *dat) {
    uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

    uint32_t skip = index % RANGE_OF_POSSIBLE_SKIPS;
    index /= RANGE_OF_POSSIBLE_SKIPS;

    if (index >= filter_5_numberOfResultsPerRun) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
    uint64_t seed = filterInput2[index].treeSeed;

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
    for (int32_t i = 0; i <= MAX_CALLS; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < FIRST_POPULATION_CHUNK.numberOfTreePositions; j++) {
            const TreeChunkPosition &tree = FIRST_POPULATION_CHUNK.treePositions[j];

            Random treeRandom(random);
            if (tree.testXZAndType(treeRandom, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version)) found |= UINT64_C(1) << j;
        }
        random.skip<1>();
    }

    if (getNumberOfOnesIn(found) != FIRST_POPULATION_CHUNK.numberOfTreePositions) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;

    uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter_6_numberOfResultsPerRun), 1);
    if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
    filterResults2[resultIndex] = Stage2Results{filterResults2[index].structureSeedIndex, seed};
#if (!CUDA_IS_PRESENT)
    return NULL;
#endif
}

#if CUDA_IS_PRESENT
__global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter_7() {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
#else
void *filter_7(void *dat) {
    uint64_t index = static_cast<ThreadData *>(dat)->index;
#endif

    if (index >= filter_6_numberOfResultsPerRun) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
    uint64_t seed = filterInput2[index].treeSeed;

    uint32_t found = 0;
    Random random = Random::withSeed(seed);
    for (int32_t i = 0; i <= MAX_CALLS; i++) {
        #pragma unroll
        for (uint32_t j = 0; j < FIRST_POPULATION_CHUNK.numberOfTreePositions; j++) {
            const TreeChunkPosition &tree = FIRST_POPULATION_CHUNK.treePositions[j];

            Random treeRandom(random);
            if (tree.testXZTypeAndAttributes(treeRandom, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version)) found |= UINT64_C(1) << j;
        }
        random.skip<1>();
    }

    if (getNumberOfOnesIn(found) != FIRST_POPULATION_CHUNK.numberOfTreePositions) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;

    uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter_7_numberOfResultsPerRun), 1);
    if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
		#if (!CUDA_IS_PRESENT)
		NULL
		#endif
		;
    filterResults2[resultIndex] = Stage2Results{filterResults2[index].structureSeedIndex, seed};
#if (!CUDA_IS_PRESENT)
    return NULL;
#endif
}

// #if CUDA_IS_PRESENT
// __global__ __launch_bounds__(ACTUAL_WORKERS_PER_BLOCK) void filter_8() {
//     uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
// #else
// void *filter_8(void *dat) {
//     uint64_t index = static_cast<ThreadData *>(dat)->index;
// #endif

//     uint32_t generated_mask = index % (UINT64_C(1) << MAX_TREE_COUNT);
//     index /= UINT64_C(1) << MAX_TREE_COUNT;

//     if (getNumberOfOnesIn(generated_mask) < FIRST_POPULATION_CHUNK.numberOfTreePositions) return
// 		#if (!CUDA_IS_PRESENT)
// 		NULL
// 		#endif
// 		;

//     if (index >= filter_7_numberOfResultsPerRun) return
// 		#if (!CUDA_IS_PRESENT)
// 		NULL
// 		#endif
// 		;
//     uint64_t seed = filterInput2[index].treeSeed;

//     Random random = Random::withSeed(seed);

//     uint32_t tree_count = biomeTreeCount(random, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version);
//     if (generated_mask >> tree_count) return
// 		#if (!CUDA_IS_PRESENT)
// 		NULL
// 		#endif
// 		;

//     uint32_t found = 0;

//     #pragma unroll
//     for (int32_t i = 0; i < tree_count; i++) {
//         bool generated = (generated_mask >> i) & 1;

//         if (generated) {
//             #pragma unroll
//             for (int32_t j = 0; j < FIRST_POPULATION_CHUNK.numberOfTreePositions; j++) {
//                 const TreeChunkPosition &tree = FIRST_POPULATION_CHUNK.treePositions[j];

//                 Random treeRandom(random);
//                 if (tree.testXZTypeAndAttributes(treeRandom, FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version)) found |= UINT64_C(1) << j;
//             }
//         }

//         TreeChunkPosition::skip(random, FIRST_POPULATION_CHUNK.biome, generated, FIRST_POPULATION_CHUNK.version);
//     }
//     if (getNumberOfOnesIn(found) != FIRST_POPULATION_CHUNK.numberOfTreePositions) return
// 		#if (!CUDA_IS_PRESENT)
// 		NULL
// 		#endif
// 		;

//     uint32_t results2_3_index = index;
//     uint32_t results2_3_mask_mask = UINT32_C(1) << (results2_3_index & 31);
//     if (atomicOr(&filter_8_masks[results2_3_index / 32], results2_3_mask_mask) & results2_3_mask_mask) return
// 		#if (!CUDA_IS_PRESENT)
// 		NULL
// 		#endif
// 		;

//     uint32_t results2_0_index = filter_8_results[index].structureSeedIndex;
//     uint32_t results2_0_mask_mask = UINT32_C(1) << (results2_0_index & 31);
//     if (atomicOr(&filter_5_masks[results2_0_index / 32], results2_0_mask_mask) & results2_0_mask_mask) return
// 		#if (!CUDA_IS_PRESENT)
// 		NULL
// 		#endif
// 		;

//     uint64_t resultIndex = atomicAdd(reinterpret_cast<unsigned long long*>(&filter_8_numberOfResultsPerRun), 1);
//     if (resultIndex >= ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) return
// 		#if (!CUDA_IS_PRESENT)
// 		NULL
// 		#endif
// 		;
//     filterResults2[resultIndex] = filter_5_results[results2_0_index];
// #if (!CUDA_IS_PRESENT)
//     return NULL;
// #endif
// }

#endif