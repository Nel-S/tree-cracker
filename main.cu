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
			
			if (theoreticalCoordsAndTypesFilterNumberOfResultsPerRun) {
				std::fprintf(stderr, "Filter 1: %" PRIu64 " results\n", oneTreeFilterNumberOfResultsPerRun);
				std::fprintf(stderr, "Filter 2: %" PRIu64 " results\n", theoreticalCoordsAndTypesFilterNumberOfResultsPerRun);
				std::fprintf(stderr, "Filter 3: %" PRIu64 " results\n", theoreticalAttributesFilterNumberOfResultsPerRun);
				std::fprintf(stderr, "Filter 4: %" PRIu64 " results\n", allAttributesFilterNumberOfResultsPerRun);
				std::fprintf(stderr, "Filter 5: %" PRIu64 " results\n", totalStructureSeedsPerRun);
				std::fprintf(stderr, "Filter 6: %" PRIu64 " results\n", totalWorldseedsPerRun);
			}
			
			oneTreeFilterNumberOfResultsPerRun = theoreticalCoordsAndTypesFilterNumberOfResultsPerRun = theoreticalAttributesFilterNumberOfResultsPerRun = allAttributesFilterNumberOfResultsPerRun = totalStructureSeedsPerRun = totalWorldseedsPerRun = 0;

			// NelS: Can this be moved?
			if (COLLAPSE_NEARBY_SEEDS_FLAG) cudaMemsetAsync(theoreticalAttributesFilterMasksPointer, 0, sizeof(theoreticalAttributesFilterMasks));

			#if CUDA_IS_PRESENT
				OneTreeFilter<<<constexprCeil(static_cast<double>(NUMBER_OF_WORKERS)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(runCurrentSeed);
			#else
				for (uint64_t i = 0; i < NUMBER_OF_WORKERS; ++i) {
					data[i].start = runCurrentSeed;
					pthread_create(&threads[i], NULL, OneTreeFilter, &data[i]);
				}
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (oneTreeFilterNumberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: One-Tree filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", oneTreeFilterNumberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, oneTreeFilterNumberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				oneTreeFilterNumberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!oneTreeFilterNumberOfResultsPerRun) continue;

			TRY_CUDA(cudaMemcpy(filterInputs, filterResults, sizeof(*filterResults)*oneTreeFilterNumberOfResultsPerRun, cudaMemcpyDefault));
			#if CUDA_IS_PRESENT
				theoreticalCoordsAndTypesFilter<<<constexprCeil(static_cast<double>(oneTreeFilterNumberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < oneTreeFilterNumberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, theoreticalCoordsAndTypesFilter, &data[i]);
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
				theoreticalAttributesFilter<<<constexprCeil(static_cast<double>(theoreticalCoordsAndTypesFilterNumberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
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
				allAttributesFilter<<<constexprCeil(static_cast<double>(theoreticalAttributesFilterNumberOfResultsPerRun * (MAX_CALLS + 1) * (UINT64_C(1) << MAX_TREE_COUNT))/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
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
				reversePopulationSeeds<<<constexprCeil(static_cast<double>(allAttributesFilterNumberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
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
			for (largeBiomesFlag = 0; largeBiomesFlag <= 1; ++largeBiomesFlag) {
				totalWorldseedsPerRun = 0;
				#if CUDA_IS_PRESENT
					biomeFilter<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun*65536)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
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