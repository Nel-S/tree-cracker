#include "src/Filters.cuh"
#include "src/ChunkRandomReversal.cuh"
#include <chrono>

int main() {
	FILE *outputFile = NULL;
	if (OUTPUT_FILEPATH) {
		outputFile = std::fopen(OUTPUT_FILEPATH, "w");
		if (!outputFile) ABORT("ERROR: Failed to open %s.\n", OUTPUT_FILEPATH);
	} else if (SILENT_MODE) ABORT("ERROR: No output method for results was provided (SILENT_MODE is enabled and no filepath was specified).\n");

	if (!SILENT_MODE) {
		POPULATION_CHUNKS_DATA.printFindings();
		constexpr double inputDataBits = POPULATION_CHUNKS_DATA.getEstimatedBits();
		if (inputDataBits < 48.) std::fprintf(stderr, "\n\nWARNING: The input data very likely does not have enough information to reduce the search space to a single structure seed (%.2g/48 bits).\nIt is VERY HIGHLY recommended you gather more data and only afterwards run the program.\n", inputDataBits);
		else {
			constexpr double inputDataHighestPopulationChunkBits = POPULATION_CHUNKS_DATA.treeChunks[0].getEstimatedBits();
			if (inputDataHighestPopulationChunkBits < 48.) std::fprintf(stderr, "\n\nWARNING: The input data's highest-information population chunk very likely does not have enough information to reduce the search space to a single structure seed by itself (%.2g/48 bits).\nOther chunks will be used later to account for this, but this program will run faster and use less memory if you gather more data and only afterwards run the program.\n", inputDataHighestPopulationChunkBits);
		}
	}

	void *theoreticalAttributesFilterMasksPointer;
	TRY_CUDA(cudaGetSymbolAddress(&theoreticalAttributesFilterMasksPointer, theoreticalAttributesFilterMasks));

	#if (!CUDA_IS_PRESENT)
		threads = static_cast<pthread_t *>(malloc(NUMBER_OF_WORKERS*sizeof(*threads)));
		__numberOfThreads = NUMBER_OF_WORKERS;
		ThreadData data[NUMBER_OF_WORKERS];
		for (uint64_t i = 0; i < NUMBER_OF_WORKERS; ++i) data[i].index = i;
	#endif

	auto start_time = std::chrono::steady_clock::now();
	for (uint64_t partialRun = ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM; partialRun <= ACTUAL_NUMBER_OF_PARTIAL_RUNS; ++partialRun) {
		// TODO: Rework to avoid overflows
		uint64_t startSeed = (ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM - 1) * (UINT64_C(1) << 44)/ACTUAL_NUMBER_OF_PARTIAL_RUNS;
		uint64_t endSeed = ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM * (UINT64_C(1) << 44)/ACTUAL_NUMBER_OF_PARTIAL_RUNS;
		if (!SILENT_MODE) std::fprintf(stderr, "Beginning population-chunk partial run %" PRIu64 "/%" PRIu64 " (states %" PRIu64 " - %" PRIu64 ").\n", partialRun, ACTUAL_NUMBER_OF_PARTIAL_RUNS, startSeed, endSeed - 1);
		for (uint64_t run_seed_start = startSeed; run_seed_start < endSeed; run_seed_start += NUMBER_OF_WORKERS) {
			if (!SILENT_MODE && TIME_PROGRAM && !(run_seed_start/NUMBER_OF_WORKERS % 256)) {
				auto last_start_time = start_time;
				start_time = std::chrono::steady_clock::now();
				double seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(start_time - last_start_time).count() / 1e9;
				double seedsPerSecond = NUMBER_OF_WORKERS / seconds;
				double eta = ((UINT64_C(1) << 44) - (run_seed_start + NUMBER_OF_WORKERS)) / seedsPerSecond;
				std::fprintf(stderr, "%.3f seconds\t%.3f billion seeds/second\tETA: %f seconds\n", seconds, seedsPerSecond/1e9, eta);
			}
			OneTreeFilterNumberOfResultsPerRun = theoreticalCoordsAndTypesFilterNumberOfResultsPerRun = theoreticalAttributesFilterNumberOfResultsPerRun = allAttributesFilterNumberOfResultsPerRun = totalStructureSeedsPerRun = totalWorldseedsPerRun = 0;

			if (COLLAPSE_NEARBY_SEEDS_FLAG) cudaMemsetAsync(theoreticalAttributesFilterMasksPointer, 0, sizeof(theoreticalAttributesFilterMasks));

			#if CUDA_IS_PRESENT
				OneTreeFilter<<<constexprCeil(static_cast<double>(NUMBER_OF_WORKERS)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(run_seed_start);
			#else
				for (uint64_t i = 0; i < NUMBER_OF_WORKERS; ++i) {
					data[i].start = run_seed_start;
					pthread_create(&threads[i], NULL, OneTreeFilter, &data[i]);
				}
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (OneTreeFilterNumberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: One-Tree filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", OneTreeFilterNumberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, OneTreeFilterNumberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				OneTreeFilterNumberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!OneTreeFilterNumberOfResultsPerRun) continue;

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*OneTreeFilterNumberOfResultsPerRun, cudaMemcpyDefault));
			#if CUDA_IS_PRESENT
				theoreticalCoordsAndTypesFilter<<<static_cast<int64_t>(ceil(static_cast<double>(OneTreeFilterNumberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK))), ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < OneTreeFilterNumberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, theoreticalCoordsAndTypesFilter, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (theoreticalCoordsAndTypesFilterNumberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Theoretical-Coords-and-Types filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", theoreticalCoordsAndTypesFilterNumberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, theoreticalCoordsAndTypesFilterNumberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				theoreticalCoordsAndTypesFilterNumberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!theoreticalCoordsAndTypesFilterNumberOfResultsPerRun) continue;

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*theoreticalCoordsAndTypesFilterNumberOfResultsPerRun, cudaMemcpyDefault));
			#if CUDA_IS_PRESENT
				theoreticalAttributesFilter<<<static_cast<int64_t>(ceil(static_cast<double>(theoreticalCoordsAndTypesFilterNumberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK))), ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < theoreticalCoordsAndTypesFilterNumberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, theoreticalAttributesFilter, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (theoreticalAttributesFilterNumberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Theoretical-Attributes filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", theoreticalAttributesFilterNumberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, theoreticalAttributesFilterNumberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				theoreticalAttributesFilterNumberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!theoreticalAttributesFilterNumberOfResultsPerRun) continue;

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*theoreticalAttributesFilterNumberOfResultsPerRun, cudaMemcpyDefault));
			#if CUDA_IS_PRESENT
				allAttributesFilter<<<static_cast<int64_t>(ceil(static_cast<double>(theoreticalAttributesFilterNumberOfResultsPerRun * (MAX_CALLS + 1) * (UINT64_C(1) << MAX_TREE_COUNT))/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK))) / ACTUAL_WORKERS_PER_BLOCK, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < theoreticalAttributesFilterNumberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, allAttributesFilter, &data[i]); // How to handle larger number of i's?
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (allAttributesFilterNumberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: All-Attributes filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", allAttributesFilterNumberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, allAttributesFilterNumberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				allAttributesFilterNumberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!allAttributesFilterNumberOfResultsPerRun) continue;

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*allAttributesFilterNumberOfResultsPerRun, cudaMemcpyDefault));
			// 1.6.4:         ?
			// 1.8.9:         ?
			// 1.12.2:        ./main input.txt output.txt 1 1.12 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 10000
			// 1.14.4:        ./main input.txt output.txt 1 1.13 60001 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 0
			// 1.16.1/1.16.4: ./main input.txt output.txt 1 1.13 80001 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 0
			#if CUDA_IS_PRESENT
				reversePopulationSeeds<<<static_cast<int64_t>(ceil(static_cast<double>(allAttributesFilterNumberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK))), ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < allAttributesFilterNumberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, reversePopulationSeeds, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (totalStructureSeedsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Population seed reverser caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalStructureSeedsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalStructureSeedsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				totalStructureSeedsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!totalStructureSeedsPerRun) continue;

			for (uint32_t i = 1; i < POPULATION_CHUNKS_DATA.numberOfTreeChunks; ++i) {
				// TODO: Second.cu filtering with other tree chunks
			}

			// TODO: Change totalStructureSeedsPerRun when second.cu filters are implemented
			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*totalStructureSeedsPerRun, cudaMemcpyDefault));
			#if CUDA_IS_PRESENT
				biomeFilter<<<static_cast<int64_t>(ceil(static_cast<double>(totalStructureSeedsPerRun*65536)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK))) / ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) pthread_create(&threads[i], NULL, biomeFilter, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (totalWorldseedsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Biome filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalWorldseedsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalWorldseedsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				totalWorldseedsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!totalWorldseedsPerRun) continue;

			for (uint64_t i = 0; i < totalWorldseedsPerRun; ++i) {
				if (OUTPUT_FILEPATH) std::fprintf(outputFile, "%" PRId64 "\n", filterResults[i]);
				if (!SILENT_MODE) std::printf("%" PRId64 "\n", filterResults[i]);
			}

			if (!SILENT_MODE && run_seed_start == startSeed) std::fprintf(stderr, "Filter counts: 1: %" PRIu64 ",\t2: %" PRIu64 ",\t3: %" PRIu64 ",\t4: %" PRIu64 "\n", OneTreeFilterNumberOfResultsPerRun, theoreticalCoordsAndTypesFilterNumberOfResultsPerRun, theoreticalAttributesFilterNumberOfResultsPerRun, allAttributesFilterNumberOfResultsPerRun);
		}
	}

	if (OUTPUT_FILEPATH) std::fclose(outputFile);

	#if (!CUDA_IS_PRESENT)
		free(threads);
	#endif
	return 0;
}