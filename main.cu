#include "src/Filters.cuh"
#include <chrono>
#include <set>

int main() {
	std::string currentFilepath = OUTPUT_FILEPATH;
	FILE *outputFile = NULL;
	std::string delimiter = "";
	// If an output filepath was specified:
	if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") {
		// First ensure the file doesn't already exist.
		// NOTE: Admittedly vulnerable to TOC/TOU.
		// (Technique adapted from PherricOxide on Stack Overflow (https://stackoverflow.com/a/12774387))
		const std::string stem = getFilepathStem(currentFilepath);
		const std::string extension = getFilepathExtension(currentFilepath);
		for (; (outputFile = std::fopen((stem + delimiter + extension).c_str(), "r")) != NULL; std::fclose(outputFile)) {
			// If it does already exist, add/increment delimiter
			if (delimiter == "") delimiter = "0";
			else delimiter = std::to_string(std::stoi(delimiter) + 1);
		}
		// Set final filepath and warn user if necessary
		currentFilepath = stem + delimiter + extension;

		outputFile = std::fopen(currentFilepath.c_str(), "w");
		if (!outputFile) ABORT("ERROR: Failed to open %s.\n", OUTPUT_FILEPATH);
	// Otherwise if an output filepath wasn't specified:
	} else if (SILENT_MODE) ABORT("ERROR: No output method for results was provided (SILENT_MODE is enabled and no filepath was specified).\n");

	if (!SILENT_MODE) {
		printSettingsAndDataWarnings();
		if (OUTPUT_FILEPATH != currentFilepath) std::fprintf(stderr, "WARNING: The specified output filepath (%s) already exists. The output file has been renamed to %s to avoid overwriting it.\n\n", OUTPUT_FILEPATH, currentFilepath.c_str());
	}

	void *filter3_masksPointer;
	TRY_CUDA(cudaGetSymbolAddress(&filter3_masksPointer, filter3_masks));

	#if (!CUDA_IS_PRESENT)
		threads = static_cast<pthread_t *>(malloc(NUMBER_OF_WORKERS*sizeof(*threads)));
		__numberOfThreads = NUMBER_OF_WORKERS;
		ThreadData data[NUMBER_OF_WORKERS];
		for (uint64_t i = 0; i < NUMBER_OF_WORKERS; ++i) data[i].index = i;
	#endif

	auto startTime = std::chrono::steady_clock::now(), currentTime = startTime;
	uint64_t runEndSeed = constexprRound(static_cast<double>(ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM - 1)*static_cast<double>(TOTAL_NUMBER_OF_STATES_TO_CHECK)/static_cast<double>(ACTUAL_NUMBER_OF_PARTIAL_RUNS));
	for (uint64_t partialRun = ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM; partialRun <= ACTUAL_NUMBER_OF_PARTIAL_RUNS; ++partialRun) {
		uint64_t runStartSeed = runEndSeed;
		runEndSeed = constexprRound(static_cast<double>(partialRun)*static_cast<double>(TOTAL_NUMBER_OF_STATES_TO_CHECK)/static_cast<double>(ACTUAL_NUMBER_OF_PARTIAL_RUNS));
		if (!SILENT_MODE) std::fprintf(stderr, "Beginning partial run #%" PRIu64 " of %" PRIu64 "\t(states [%" PRIu64 ", %" PRIu64 "] out of %" PRIu64 ").\n", partialRun, ACTUAL_NUMBER_OF_PARTIAL_RUNS, runStartSeed, runEndSeed - 1, TOTAL_NUMBER_OF_STATES_TO_CHECK - 1);
		for (uint64_t runCurrentSeed = runStartSeed; runCurrentSeed < runEndSeed; runCurrentSeed += NUMBER_OF_WORKERS) {
			if (!SILENT_MODE && PRINT_TIMESTAMPS_FREQUENCY && !(runCurrentSeed/NUMBER_OF_WORKERS % PRINT_TIMESTAMPS_FREQUENCY)) {
				auto lastTime = currentTime;
				currentTime = std::chrono::steady_clock::now();
				double totalSeconds = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime).count()/1e9;
				if (totalSeconds > 0.2) { // So we don't get a timestamp with meaningless values for the very first, 0-second entry
					double durationSeconds = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - lastTime).count()/1e9;
					double seedsPerSecond = (NUMBER_OF_WORKERS*PRINT_TIMESTAMPS_FREQUENCY)/durationSeconds;
					double eta = (TOTAL_NUMBER_OF_STATES_TO_CHECK - (runCurrentSeed + NUMBER_OF_WORKERS))/seedsPerSecond;
					std::fprintf(stderr, "%.3f seconds\t%.3f billion seeds/second\tETA: %.3f seconds\n", totalSeconds, seedsPerSecond/1e9, eta);
				}
			}
			
			currentPopulationChunkDataIndex = 0;

			if (ABSOLUTE_POPULATION_CHUNKS_DATA.collapseNearbySeedsFlag) cudaMemsetAsync(filter3_masksPointer, 0, sizeof(filter3_masks));

			filter1_numberOfResultsPerRun = 0;
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

			
			if (static_cast<void *>(FILTER_2_INPUT) != static_cast<void *>(FILTER_1_OUTPUT)) {
				#if CUDA_IS_PRESENT
					// transferResults<<<constexprCeil(static_cast<double>(filter1_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_1_OUTPUT, FILTER_2_INPUT);
					// transferResults<<<filter1_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_1_OUTPUT, FILTER_2_INPUT);
				#else
					for (uint64_t i = 0; i < filter1_numberOfResultsPerRun; ++i) {
						pthread_create(&threads[i], NULL, transferResults, &data[i]);
					}
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());
			}

			filter2_numberOfResultsPerRun = 0;
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


			if (static_cast<void *>(FILTER_3_INPUT) != static_cast<void *>(FILTER_2_OUTPUT)) {
				#if CUDA_IS_PRESENT
					// transferResults<<<constexprCeil(static_cast<double>(filter2_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_2_OUTPUT, FILTER_3_INPUT);
					// transferResults<<<filter2_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_2_OUTPUT, FILTER_3_INPUT);
				#else
					for (uint64_t i = 0; i < filter2_numberOfResultsPerRun; ++i) {
						pthread_create(&threads[i], NULL, transferResults, &data[i]);
					}
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());
			}

			filter3_numberOfResultsPerRun = 0;
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


			if (static_cast<void *>(TREECHUNK_FILTER_INPUT) != static_cast<void *>(FILTER_3_OUTPUT)) {
				#if CUDA_IS_PRESENT
					// transferResults<<<constexprCeil(static_cast<double>(filter3_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_3_OUTPUT, TREECHUNK_FILTER_INPUT);
					// transferResults<<<filter3_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_3_OUTPUT, TREECHUNK_FILTER_INPUT);
				#else
					for (uint64_t i = 0; i < filter3_numberOfResultsPerRun; ++i) {
						pthread_create(&threads[i], NULL, transferResults, &data[i]);
					}
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());
			}

			treechunkFilter_numberOfResultsPerRun = 0;
			#if CUDA_IS_PRESENT
				// treechunkFilter<<<constexprCeil(static_cast<double>(constexprMin(filter3_numberOfResultsPerRun * (ABSOLUTE_POPULATION_CHUNKS_DATA.getCurrentMaxCalls() + 1) * (UINT64_C(1) << ABSOLUTE_POPULATION_CHUNKS_DATA.getCurrentMaxTreeCount()))/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
				treechunkFilter<<<constexprMin(filter3_numberOfResultsPerRun * (ABSOLUTE_POPULATION_CHUNKS_DATA.getCurrentMaxCalls() + 1) * (UINT64_C(1) << ABSOLUTE_POPULATION_CHUNKS_DATA.getCurrentMaxTreeCount())/ACTUAL_WORKERS_PER_BLOCK + 1, static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
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

			#if (!DEBUG)
			if (static_cast<void *>(POPULATION_REVERSAL_INPUT) != static_cast<void *>(TREECHUNK_FILTER_OUTPUT)) {
				#if CUDA_IS_PRESENT
					// transferResults<<<constexprCeil(static_cast<double>(treechunkFilter_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(TREECHUNK_FILTER_OUTPUT, POPULATION_REVERSAL_INPUT);
					// transferResults<<<treechunkFilter_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(TREECHUNK_FILTER_OUTPUT, POPULATION_REVERSAL_INPUT);
				#else
					for (uint64_t i = 0; i < treechunkFilter_numberOfResultsPerRun; ++i) {
						pthread_create(&threads[i], NULL, transferResults, &data[i]);
					}
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());
			}
			
			// 1.6.4:         ?
			// 1.8.9:         ?
			// 1.12.2:        ./main input.txt output.txt 1 1.12 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 10000
			// 1.14.4:        ./main input.txt output.txt 1 1.13 60001 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 0
			// 1.16.1/1.16.4: ./main input.txt output.txt 1 1.13 80001 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 0
			totalStructureSeedsPerRun = 0;
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

			for (currentPopulationChunkDataIndex = 1; currentPopulationChunkDataIndex < ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks; ++currentPopulationChunkDataIndex) {
				if (static_cast<void *>(FILTER_5_INPUT) != (currentPopulationChunkDataIndex == 1 ? static_cast<void *>(POPULATION_REVERSAL_OUTPUT) : static_cast<void *>(FILTER_8_OUTPUT))) {
					// #if CUDA_IS_PRESENT
					// 	// transferResults<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_5_INPUT);
					// 	transferResults<<<totalStructureSeedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_5_INPUT);
						
					// #else
					// 	for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
					// 		pthread_create(&threads[i], NULL, transferResults, &data[i]);
					// 	}
					// #endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
					TRY_CUDA(cudaMemcpy(FILTER_5_INPUT, currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(currentPopulationChunkDataIndex == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), cudaMemcpyKind::cudaMemcpyDefault));
				}

				filter5_numberOfResultsPerRun = 0;
				#if CUDA_IS_PRESENT
					// filter5<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					filter5<<<totalStructureSeedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) pthread_create(&threads[i], NULL, filter5, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (filter5_numberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Structure-to-treechunk Filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter5_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter5_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					filter5_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!filter5_numberOfResultsPerRun) continue;

				
				if (static_cast<void *>(FILTER_6_INPUT) != static_cast<void *>(FILTER_5_OUTPUT)) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(filter5_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_5_OUTPUT, FILTER_6_INPUT);
						// transferResults<<<filter5_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_5_OUTPUT, FILTER_6_INPUT);
					#else
						for (uint64_t i = 0; i < filter5_numberOfResultsPerRun; ++i) {
							pthread_create(&threads[i], NULL, transferResults, &data[i]);
						}
					#endif
					TRY_CUDA(cudaGetLastError());
					TRY_CUDA(cudaDeviceSynchronize());
				}

				filter6_numberOfResultsPerRun = 0;
				#if CUDA_IS_PRESENT
					// filter6<<<constexprCeil(static_cast<double>(filter5_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					filter6<<<filter5_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < filter5_numberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, filter6, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (filter6_numberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Structure-to-treechunk Filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter6_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter6_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					filter6_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!filter6_numberOfResultsPerRun) continue;


				if (static_cast<void *>(FILTER_7_INPUT) != static_cast<void *>(FILTER_6_OUTPUT)) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(filter6_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_6_OUTPUT, FILTER_7_INPUT);
						// transferResults<<<filter6_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_6_OUTPUT, FILTER_7_INPUT);
					#else
						for (uint64_t i = 0; i < filter6_numberOfResultsPerRun; ++i) {
							pthread_create(&threads[i], NULL, transferResults, &data[i]);
						}
					#endif
					TRY_CUDA(cudaGetLastError());
					TRY_CUDA(cudaDeviceSynchronize());
				}

				filter7_numberOfResultsPerRun = 0;
				#if CUDA_IS_PRESENT
					// filter7<<<constexprCeil(static_cast<double>(filter6_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					filter7<<<filter6_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < filter6_numberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, filter7, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (filter7_numberOfResultsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Structure-to-treechunk Filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter7_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter7_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					filter7_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!filter7_numberOfResultsPerRun) continue;


				if (static_cast<void *>(FILTER_8_STRUCTURESEED_INPUT) != (currentPopulationChunkDataIndex == 1 ? static_cast<void *>(POPULATION_REVERSAL_OUTPUT) : static_cast<void *>(FILTER_8_OUTPUT))) {
					// #if CUDA_IS_PRESENT
					// 	// transferResults<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_8_STRUCTURESEED_INPUT);
					// 	transferResults<<<totalStructureSeedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_8_STRUCTURESEED_INPUT);
					// #else
					// 	for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
					// 		pthread_create(&threads[i], NULL, transferResults, &data[i]);
					// 	}
					// #endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
					TRY_CUDA(cudaMemcpy(FILTER_8_STRUCTURESEED_INPUT, currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(currentPopulationChunkDataIndex == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), cudaMemcpyKind::cudaMemcpyDefault));
				}
				if (static_cast<void *>(FILTER_8_TREECHUNK_INPUT) != static_cast<void *>(FILTER_7_OUTPUT)) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(filter7_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_7_OUTPUT, FILTER_8_TREECHUNK_INPUT);
						// transferResults<<<filter7_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_7_OUTPUT, FILTER_8_TREECHUNK_INPUT);
					#else
						for (uint64_t i = 0; i < filter7_numberOfResultsPerRun; ++i) {
							pthread_create(&threads[i], NULL, transferResults, &data[i]);
						}
					#endif
					TRY_CUDA(cudaGetLastError());
					TRY_CUDA(cudaDeviceSynchronize());
				}

				totalStructureSeedsPerRun = 0;
				#if CUDA_IS_PRESENT
					// filter8<<<constexprCeil(static_cast<double>(filter7_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					filter8<<<filter7_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < filter7_numberOfResultsPerRun; ++i) pthread_create(&threads[i], NULL, filter8, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (totalStructureSeedsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Filter 8 caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalStructureSeedsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalStructureSeedsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					totalStructureSeedsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!totalStructureSeedsPerRun) continue;
			}

			// Pass all structure seeds through a set to remove duplicates (and also order them)
			std::set<uint64_t> uniqueStructureSeeds;
			for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) uniqueStructureSeeds.insert(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT[i] : FILTER_8_OUTPUT[i]);
			totalStructureSeedsPerRun = static_cast<uint64_t>(uniqueStructureSeeds.size());
			uint64_t count = 0;
			for (auto i = uniqueStructureSeeds.cbegin(); i != uniqueStructureSeeds.cend(); ++i) (ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT[count++] : FILTER_8_OUTPUT[count++]) = *i;
			uniqueStructureSeeds.clear();
			#endif

			// if (static_cast<void *>(WORLDSEED_FILTER_INPUT) != (ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? static_cast<void *>(POPULATION_REVERSAL_OUTPUT) : static_cast<void *>(FILTER_8_OUTPUT)))
			// #if CUDA_IS_PRESENT
			// 	// transferResults<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, WORLDSEED_FILTER_INPUT);
			// 	transferResults<<<totalStructureSeedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, WORLDSEED_FILTER_INPUT);
			// #else
			// 	for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
			// 		pthread_create(&threads[i], NULL, transferResults, &data[i]);
			// 	}
			// #endif
			// TRY_CUDA(cudaGetLastError());
			// TRY_CUDA(cudaDeviceSynchronize());

			// for (largeBiomesFlag = 0; largeBiomesFlag <= 1; ++largeBiomesFlag) {
			// 	totalWorldseedsPerRun = 0;
			// 	#if CUDA_IS_PRESENT
			// 		// biomeFilter<<<constexprMin(constexprCeil(static_cast<double>(totalStructureSeedsPerRun*65536)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
			// 		biomeFilter<<<constexprMin(totalStructureSeedsPerRun*65536/ACTUAL_WORKERS_PER_BLOCK + 1, static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
			// 	#else
			// 		for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) pthread_create(&threads[i], NULL, biomeFilter, &data[i]);
			// 	#endif
			// 	TRY_CUDA(cudaGetLastError());
			// 	TRY_CUDA(cudaDeviceSynchronize());

			// 	if (totalWorldseedsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
			// 		if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Biome filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalWorldseedsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalWorldseedsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
			// 		totalWorldseedsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			// 	}
			// 	for (uint64_t i = 0; i < totalWorldseedsPerRun; ++i) {
				// const char *notes = getNotesAboutSeed(WORLDSEED_FILTER_OUTPUT[i], largeBiomesFlag);
				// if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") std::fprintf(outputFile, "%" PRId64"\t%s\n", static_cast<int64_t>(WORLDSEED_FILTER_OUTPUT[i]), notes);
				// if (!SILENT_MODE) std::printf("%" PRId64 "\t%s\n", static_cast<int64_t>(WORLDSEED_FILTER_OUTPUT[i]), notes);
			// }

			#if DEBUG
			for (uint64_t i = 0; i < treechunkFilter_numberOfResultsPerRun; ++i) {
				if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") std::fprintf(outputFile, "%" PRId64"\n", static_cast<int64_t>(TREECHUNK_FILTER_OUTPUT[i]));
				if (!SILENT_MODE) std::printf("%" PRId64 "\n", static_cast<int64_t>(TREECHUNK_FILTER_OUTPUT[i]));
			}
			#else
			for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
				if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") std::fprintf(outputFile, "%" PRId64"\n", static_cast<int64_t>(POPULATION_REVERSAL_OUTPUT[i]));
				if (!SILENT_MODE) std::printf("%" PRId64 "\n", static_cast<int64_t>(POPULATION_REVERSAL_OUTPUT[i]));
			}
			#endif
		}
	}

	if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") std::fclose(outputFile);

	#if (!CUDA_IS_PRESENT)
		free(threads);
	#endif
	return 0;
}