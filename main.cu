#include "src/Filters.cuh"
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
		if (inputDataBits < 48.) std::fprintf(stderr, "WARNING: The input data very likely does not have enough information to reduce the search space to a single structure seed (%.2g/48 bits).\nIt is VERY HIGHLY recommended you gather more data and only afterwards run the program.\n\n", inputDataBits);
		else {
			constexpr double inputDataHighestPopulationChunkBits = POPULATION_CHUNKS_DATA.treeChunks[0].getEstimatedBits();
			if (inputDataHighestPopulationChunkBits < 48.) std::fprintf(stderr, "\n\nWARNING: The input data's highest-information population chunk very likely does not have enough information to reduce the search space to a single structure seed by itself (%.2g/48 bits).\nOther chunks will be used later to account for this, but this program will run faster and use less memory if you gather more data and only afterwards run the program.\n", inputDataHighestPopulationChunkBits);
		}
	}

	void *theoreticalAttributesFilterMasksPointer;
	TRY_CUDA(cudaGetSymbolAddress(&theoreticalAttributesFilterMasksPointer, filter3_masks));

	#if (!CUDA_IS_PRESENT)
		threads = static_cast<pthread_t *>(malloc(NUMBER_OF_WORKERS*sizeof(*threads)));
		__numberOfThreads = NUMBER_OF_WORKERS;
		ThreadData data[NUMBER_OF_WORKERS];
		for (uint64_t i = 0; i < NUMBER_OF_WORKERS; ++i) data[i].index = i;
	#endif

	auto startTime = std::chrono::steady_clock::now(), currentTime = startTime;
	for (uint64_t partialRun = ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM; partialRun <= ACTUAL_NUMBER_OF_PARTIAL_RUNS; ++partialRun) {
		// TODO: Rework to avoid overflows
		uint64_t runStartSeed = constexprRound(static_cast<double>((partialRun - 1)*TOTAL_NUMBER_OF_STATES_TO_CHECK)/static_cast<double>(ACTUAL_NUMBER_OF_PARTIAL_RUNS));
		uint64_t runEndSeed = constexprRound(static_cast<double>(partialRun*TOTAL_NUMBER_OF_STATES_TO_CHECK)/static_cast<double>(ACTUAL_NUMBER_OF_PARTIAL_RUNS));
		if (!SILENT_MODE) std::fprintf(stderr, "Beginning partial run #%" PRIu64 " of %" PRIu64 "\t(states [%" PRIu64 ", %" PRIu64 "] out of %" PRIu64 ").\n", partialRun, ACTUAL_NUMBER_OF_PARTIAL_RUNS, runStartSeed, runEndSeed - 1, TOTAL_NUMBER_OF_STATES_TO_CHECK - 1);
		for (uint64_t runCurrentSeed = runStartSeed; runCurrentSeed < runEndSeed; runCurrentSeed += NUMBER_OF_WORKERS) {
			if (!SILENT_MODE && TIME_PROGRAM && !(runCurrentSeed/NUMBER_OF_WORKERS & 255)) {
				auto lastTime = currentTime;
				currentTime = std::chrono::steady_clock::now();
				double totalSeconds = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count()/1e9;
				if (totalSeconds > 0.2) { // So we don't get a timestamp with meaningless values for the very first, 0-second entry
					double durationSeconds = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - lastTime).count()/1e9;
					double seedsPerSecond = NUMBER_OF_WORKERS/durationSeconds;
					double eta = (TOTAL_NUMBER_OF_STATES_TO_CHECK - (runCurrentSeed + NUMBER_OF_WORKERS))/seedsPerSecond;
					std::fprintf(stderr, "%.3f seconds\t%.3f billion seeds/second\tETA: %f seconds\n", totalSeconds, seedsPerSecond/1e9, eta);
				}
			}
			
			if (filter2_numberOfResultsPerRun) {
				std::fprintf(stderr, "Filter 1: %" PRIu64 " results\n", filter1_numberOfResultsPerRun);
				std::fprintf(stderr, "Filter 2: %" PRIu64 " results\n", filter2_numberOfResultsPerRun);
				std::fprintf(stderr, "Filter 3: %" PRIu64 " results\n", filter3_numberOfResultsPerRun);
				std::fprintf(stderr, "Filter 4: %" PRIu64 " results\n", treechunkFilter_numberOfResultsPerRun);
				std::fprintf(stderr, "Filter 5: %" PRIu64 " results\n", totalStructureSeedsPerRun);
				std::fprintf(stderr, "Filter 6: %" PRIu64 " results\n", totalWorldseedsPerRun);
			}
			
			filter1_numberOfResultsPerRun = filter2_numberOfResultsPerRun = filter3_numberOfResultsPerRun = treechunkFilter_numberOfResultsPerRun = totalStructureSeedsPerRun = totalWorldseedsPerRun = 0;
			currentPopulationChunkDataIndex = 0;

			// NelS: Can this be moved?
			if (COLLAPSE_NEARBY_SEEDS_FLAG) cudaMemsetAsync(theoreticalAttributesFilterMasksPointer, 0, sizeof(filter3_masks));

			#if CUDA_IS_PRESENT
				// filter1<<<constexprCeil(static_cast<double>(NUMBER_OF_WORKERS)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(runCurrentSeed);
				filter1<<<NUMBER_OF_WORKERS/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(runCurrentSeed);
			#else
				for (uint64_t i = 0; i < NUMBER_OF_WORKERS; ++i) {
					data[i].start = runCurrentSeed;
					pthread_create(&threads[i], NULL, filter1, &data[i]);
				}
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (filter1_numberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Filter 1 caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter1_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter1_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				filter1_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!filter1_numberOfResultsPerRun) continue;

			// for (uint64_t i = 0; i < filter1_numberOfResultsPerRun; ++i) {

			// }

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*filter1_numberOfResultsPerRun, cudaMemcpyDefault));
			#if CUDA_IS_PRESENT
				// filter2<<<constexprCeil(static_cast<double>(filter1_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
				filter2<<<filter1_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < filter1_numberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, filter2, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (filter2_numberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Theoretical-Coords-and-Types filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter2_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter2_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				filter2_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!filter2_numberOfResultsPerRun) continue;

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*filter2_numberOfResultsPerRun, cudaMemcpyDefault));
			#if CUDA_IS_PRESENT
				// filter3<<<constexprCeil(static_cast<double>(filter2_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
				filter3<<<filter2_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < filter2_numberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, filter3, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (filter3_numberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Theoretical-Attributes filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter3_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter3_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				filter3_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!filter3_numberOfResultsPerRun) continue;

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*filter3_numberOfResultsPerRun, cudaMemcpyDefault));
			#if CUDA_IS_PRESENT
				// treechunkFilter<<<constexprCeil(static_cast<double>(filter3_numberOfResultsPerRun * (MAX_CALLS + 1) * (UINT64_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount))/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
				treechunkFilter<<<filter3_numberOfResultsPerRun * (MAX_CALLS + 1) * (UINT64_C(1) << POPULATION_CHUNKS_DATA.treeChunks[currentPopulationChunkDataIndex].maxTreeCount)/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < filter3_numberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, treechunkFilter, &data[i]); // How to handle larger number of i's?
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (treechunkFilter_numberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: All-Attributes filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", treechunkFilter_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, treechunkFilter_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				treechunkFilter_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!treechunkFilter_numberOfResultsPerRun) continue;

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*treechunkFilter_numberOfResultsPerRun, cudaMemcpyDefault));
			// 1.6.4:         ?
			// 1.8.9:         ?
			// 1.12.2:        ./main input.txt output.txt 1 1.12 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 10000
			// 1.14.4:        ./main input.txt output.txt 1 1.13 60001 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 0
			// 1.16.1/1.16.4: ./main input.txt output.txt 1 1.13 80001 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 0
			#if CUDA_IS_PRESENT
				// reversePopulationSeeds<<<constexprCeil(static_cast<double>(treechunkFilter_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
				reversePopulationSeeds<<<treechunkFilter_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < treechunkFilter_numberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, reversePopulationSeeds, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (totalStructureSeedsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Population seed reverser caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalStructureSeedsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalStructureSeedsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				totalStructureSeedsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!totalStructureSeedsPerRun) continue;

			for (currentPopulationChunkDataIndex = 1; currentPopulationChunkDataIndex < POPULATION_CHUNKS_DATA.numberOfTreeChunks; ++currentPopulationChunkDataIndex) {
				// TODO: Second.cu filtering with other tree chunks
			}

			// TODO: Change totalStructureSeedsPerRun when second.cu filters are implemented
			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*totalStructureSeedsPerRun, cudaMemcpyDefault));
			for (largeBiomesFlag = 0; largeBiomesFlag <= 1; ++largeBiomesFlag) {
				totalWorldseedsPerRun = 0;
				#if CUDA_IS_PRESENT
					// biomeFilter<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun*65536)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					biomeFilter<<<totalStructureSeedsPerRun*65536/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) pthread_create(&threads[i], NULL, biomeFilter, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (totalWorldseedsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Biome filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalWorldseedsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalWorldseedsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					totalWorldseedsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (totalWorldseedsPerRun) {
					for (uint64_t i = 0; i < totalWorldseedsPerRun; ++i) {
						const char *notes = getNotesAboutSeed(filterResults[i], largeBiomesFlag);
						if (OUTPUT_FILEPATH) std::fprintf(outputFile, "%" PRId64 "%s\n", static_cast<int64_t>(filterResults[i]), notes);
						if (!SILENT_MODE) std::printf("%" PRId64 "%s\n", static_cast<int64_t>(filterResults[i]), notes);
					}
				}
			}
		}
	}

	if (OUTPUT_FILEPATH) std::fclose(outputFile);

	#if (!CUDA_IS_PRESENT)
		free(threads);
	#endif
	return 0;
}