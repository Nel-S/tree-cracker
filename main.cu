#include "src/Filters.cuh"
#include <chrono>
#include <set>

int main() {
	std::string actualFilepath = OUTPUT_FILEPATH;
	FILE *outputFile = NULL;
	std::string delimiter = "";
	// If an output filepath was specified:
	if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") {
		// First ensure the file doesn't already exist.
		// NOTE: Admittedly vulnerable to TOC/TOU.
		// (Technique adapted from PherricOxide on Stack Overflow (https://stackoverflow.com/a/12774387))
		const std::string stem = getFilepathStem(actualFilepath);
		const std::string extension = getFilepathExtension(actualFilepath);
		for (; (outputFile = std::fopen((stem + delimiter + extension).c_str(), "r")) != NULL; std::fclose(outputFile)) {
			// If it does already exist, add/increment delimiter
			// TODO: Check if original filename ends with a number, and if so increment that directly
			if (delimiter == "") delimiter = "0";
			else delimiter = std::to_string(std::stoi(delimiter) + 1);
		}
		// Set final filepath and warn user if delimiter was needed
		actualFilepath = stem + delimiter + extension;

		outputFile = std::fopen(actualFilepath.c_str(), "w");
		if (!outputFile) ABORT("ERROR: Failed to open %s.\n", OUTPUT_FILEPATH);
	// Otherwise if an output filepath wasn't specified: abort if results won't be printed to the screen either
	} else if (SILENT_MODE) ABORT("ERROR: No output method for results was provided (SILENT_MODE is enabled and no filepath was specified).\n");

	if (!SILENT_MODE) {
		// Print input structure and setting changes
		printSettingsAndDataWarnings();
		// Also warn if the filepath was changed
		if (OUTPUT_FILEPATH != actualFilepath) std::fprintf(stderr, "WARNING: The specified output filepath (%s) already exists. The output file has been renamed to %s to avoid overwriting it.\n\n", OUTPUT_FILEPATH, actualFilepath.c_str());
	}

	void *filter3_masksPointer;
	TRY_CUDA(cudaGetSymbolAddress(&filter3_masksPointer, filter3_masks));

	// Sets up 
	#if (!CUDA_IS_PRESENT)
		threads = reinterpret_cast<pthread_t *>(malloc(NUMBER_OF_WORKERS*sizeof(*threads)));
		__numberOfThreads = NUMBER_OF_WORKERS;
		ThreadData data[NUMBER_OF_WORKERS];
		for (uint64_t i = 0; i < NUMBER_OF_WORKERS; ++i) data[i].index = i;
	#endif

	auto startTime = std::chrono::steady_clock::now(), currentTime = startTime;
	uint64_t runEndSeed = constexprRound(static_cast<double>(ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM - 1)*static_cast<double>(TOTAL_NUMBER_OF_STATES_TO_CHECK)/static_cast<double>(ACTUAL_NUMBER_OF_PARTIAL_RUNS));
	for (uint64_t partialRun = ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM; partialRun <= ACTUAL_NUMBER_OF_PARTIAL_RUNS; ++partialRun) {
		uint64_t runStartSeed = runEndSeed;
		runEndSeed = constexprRound(static_cast<double>(partialRun)*static_cast<double>(TOTAL_NUMBER_OF_STATES_TO_CHECK)/static_cast<double>(ACTUAL_NUMBER_OF_PARTIAL_RUNS));
		if (!SILENT_MODE) {
			if (partialRun != ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM) {
				std::fprintf(stderr, "(Filter 1: %" PRIu64 " results, \tFilter 2: %" PRIu64 " results, \tFilter 3: %" PRIu64 " results, \tFilter 4: %" PRIu64 " results", filter1_numberOfResultsPerRun, filter2_numberOfResultsPerRun, filter3_numberOfResultsPerRun, treechunkFilter_numberOfResultsPerRun);
				filter1_numberOfResultsPerRun = filter2_numberOfResultsPerRun = filter3_numberOfResultsPerRun = treechunkFilter_numberOfResultsPerRun = 0;
				if (static_cast<int32_t>(ACTUAL_TYPES_TO_OUTPUT) > 2*static_cast<int32_t>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds) - 1) {
					std::fprintf(stderr, ", \tPopulation Chunk Reversal: %" PRIu64 " results", ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? totalStructureSeedsPerRun : filter5_numberOfResultsPerRun);
					if (1 < ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks) {
						std::fprintf(stderr, "\n Filter 5: %" PRIu64 " results, \tFilter 6: %" PRIu64 " results, \tFilter 7: %" PRIu64 " results, \tFilter 8: %" PRIu64 " results", filter5_numberOfResultsPerRun, filter6_numberOfResultsPerRun, filter7_numberOfResultsPerRun, totalStructureSeedsPerRun);
						filter5_numberOfResultsPerRun = filter6_numberOfResultsPerRun = filter7_numberOfResultsPerRun = 0;
					}
					totalStructureSeedsPerRun = 0;
					if (static_cast<int32_t>(ACTUAL_TYPES_TO_OUTPUT) > 2*static_cast<int32_t>(OutputType::Structure_Seeds) - 1) {
						std::fprintf(stderr, ", \tBiome Filter: %" PRIu64 " results", totalWorldseedsPerRun);
						totalWorldseedsPerRun = 0;
					}
				}
				std::fprintf(stderr, ")\n\n");
			}
			std::fprintf(stderr, "Beginning partial run #%" PRIu64 " of %" PRIu64 "\t(states [%" PRIu64 ", %" PRIu64 "] out of %" PRIu64 ").\n", partialRun, ACTUAL_NUMBER_OF_PARTIAL_RUNS, runStartSeed, runEndSeed - 1, TOTAL_NUMBER_OF_STATES_TO_CHECK - 1);
		}
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
			
			// Begin with just focusing on chunk 0
			currentPopulationChunkDataIndex = 0;

			if (ABSOLUTE_POPULATION_CHUNKS_DATA.collapseNearbySeedsFlag) cudaMemsetAsync(filter3_masksPointer, 0, sizeof(filter3_masks));

			// Filter 1 (states that can exactly generate the highest-information tree):
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
					TRY_CUDA(cudaMemcpy(FILTER_2_INPUT, FILTER_1_OUTPUT, filter1_numberOfResultsPerRun*sizeof(*FILTER_1_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
				#else
					// for (uint64_t i = 0; i < filter1_numberOfResultsPerRun; ++i) {
					// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
					// }
					if (!memcpy(FILTER_2_INPUT, FILTER_1_OUTPUT, filter1_numberOfResultsPerRun*sizeof(*FILTER_1_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of FILTER_1_OUTPUT to FILTER_2_INPUT.\n", filter1_numberOfResultsPerRun*static_cast<uint64_t>(sizeof(*FILTER_1_OUTPUT)));
				#endif
				// TRY_CUDA(cudaGetLastError());
				// TRY_CUDA(cudaDeviceSynchronize());
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
					TRY_CUDA(cudaMemcpy(FILTER_3_INPUT, FILTER_2_OUTPUT, filter2_numberOfResultsPerRun*sizeof(*FILTER_2_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
				#else
					// for (uint64_t i = 0; i < filter2_numberOfResultsPerRun; ++i) {
					// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
					// }
					if (!memcpy(FILTER_3_INPUT, FILTER_2_OUTPUT, filter2_numberOfResultsPerRun*sizeof(*FILTER_2_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of FILTER_2_OUTPUT to FILTER_3_INPUT.\n", filter2_numberOfResultsPerRun*static_cast<uint64_t>(sizeof(*FILTER_2_OUTPUT)));
				#endif
				// TRY_CUDA(cudaGetLastError());
				// TRY_CUDA(cudaDeviceSynchronize());
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
					TRY_CUDA(cudaMemcpy(TREECHUNK_FILTER_INPUT, FILTER_3_OUTPUT, filter3_numberOfResultsPerRun*sizeof(*FILTER_3_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
				#else
					// for (uint64_t i = 0; i < filter3_numberOfResultsPerRun; ++i) {
					// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
					// }
					if (!memcpy(TREECHUNK_FILTER_INPUT, FILTER_3_OUTPUT, filter3_numberOfResultsPerRun*sizeof(*FILTER_3_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of FILTER_3_OUTPUT to TREECHUNK_FILTER_INPUT.\n", filter3_numberOfResultsPerRun*static_cast<uint64_t>(sizeof(*FILTER_3_OUTPUT)));
				#endif
				// TRY_CUDA(cudaGetLastError());
				// TRY_CUDA(cudaDeviceSynchronize());
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

			if (static_cast<bool>(ACTUAL_TYPES_TO_OUTPUT & static_cast<OutputType>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds))) {
				if (static_cast<void *>(PRINT_INPUT) != static_cast<void *>(TREECHUNK_FILTER_OUTPUT)) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(treechunkFilter_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(TREECHUNK_FILTER_OUTPUT, PRINT_INPUT);
						// transferResults<<<treechunkFilter_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(TREECHUNK_FILTER_OUTPUT, PRINT_INPUT);
						// TRY_CUDA(cudaMemcpy(PRINT_INPUT, TREECHUNK_FILTER_OUTPUT, treechunkFilter_numberOfResultsPerRun*sizeof(*TREECHUNK_FILTER_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
						// TODO: This is failing to transfer the results properly, causing all printed treechunk seeds to be 0. Why?
						// TODO: Maybe only print here if Structure_Seeds etc. are disabled; otherwise bundle/derive treechunk seeds with structure seeds/worldseeds at later print
						TRY_CUDA(cudaMemcpy(PRINT_INPUT, TREECHUNK_FILTER_OUTPUT, treechunkFilter_numberOfResultsPerRun*sizeof(*TREECHUNK_FILTER_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
					#else
						// for (uint64_t i = 0; i < treechunkFilter_numberOfResultsPerRun; ++i) {
						// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
						// }
						if (!memcpy(PRINT_INPUT, TREECHUNK_FILTER_OUTPUT, treechunkFilter_numberOfResultsPerRun*sizeof(*TREECHUNK_FILTER_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of TREECHUNK_FILTER_OUTPUT to PRINT_INPUT.\n", treechunkFilter_numberOfResultsPerRun*static_cast<uint64_t>(sizeof(*TREECHUNK_FILTER_OUTPUT)));
					#endif
					TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
				}

				for (uint64_t i = 0; i < treechunkFilter_numberOfResultsPerRun; ++i) {
					// TODO: Possibly replace with toString(OutputType) when implemented?
					if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") {
						std::fprintf(outputFile, "%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), ACTUAL_TYPES_TO_OUTPUT == static_cast<OutputType>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds) ? "" : "\t(Highest Information Treechunk seed)");
						std::fflush(outputFile);
					}
					if (!SILENT_MODE) std::printf("%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), ACTUAL_TYPES_TO_OUTPUT == static_cast<OutputType>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds) ? "" : "\t(Highest Information Treechunk seed)");
				}
				if (static_cast<int32_t>(ACTUAL_TYPES_TO_OUTPUT) <= 2*static_cast<int32_t>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds) - 1) continue;
			}

			if (static_cast<void *>(POPULATION_REVERSAL_INPUT) != static_cast<void *>(TREECHUNK_FILTER_OUTPUT)) {
				#if CUDA_IS_PRESENT
					// transferResults<<<constexprCeil(static_cast<double>(treechunkFilter_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(TREECHUNK_FILTER_OUTPUT, POPULATION_REVERSAL_INPUT);
					// transferResults<<<treechunkFilter_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(TREECHUNK_FILTER_OUTPUT, POPULATION_REVERSAL_INPUT);
					TRY_CUDA(cudaMemcpy(POPULATION_REVERSAL_INPUT, TREECHUNK_FILTER_OUTPUT, treechunkFilter_numberOfResultsPerRun*sizeof(*TREECHUNK_FILTER_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
				#else
					// for (uint64_t i = 0; i < treechunkFilter_numberOfResultsPerRun; ++i) {
					// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
					// }
					if (!memcpy(POPULATION_REVERSAL_INPUT, TREECHUNK_FILTER_OUTPUT, treechunkFilter_numberOfResultsPerRun*sizeof(*TREECHUNK_FILTER_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of TREECHUNK_FILTER_OUTPUT to POPULATION_REVERSAL_INPUT.\n", treechunkFilter_numberOfResultsPerRun*static_cast<uint64_t>(sizeof(*TREECHUNK_FILTER_OUTPUT)));
				#endif
				// TRY_CUDA(cudaGetLastError());
				// TRY_CUDA(cudaDeviceSynchronize());
			}
			
			// From Epic10l2
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
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_5_INPUT);
						// transferResults<<<totalStructureSeedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_5_INPUT);
						TRY_CUDA(cudaMemcpy(FILTER_5_INPUT, currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(currentPopulationChunkDataIndex == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), cudaMemcpyKind::cudaMemcpyDefault));
					#else
						// for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
						// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
						// }
						if (!memcpy(FILTER_5_INPUT, currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(currentPopulationChunkDataIndex == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of %s to FILTER_5_INPUT.\n", totalStructureSeedsPerRun*(currentPopulationChunkDataIndex == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), currentPopulationChunkDataIndex == 1 ? "POPULATION_REVERSAL_OUTPUT" : "FILTER_8_OUTPUT");
					#endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
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
						TRY_CUDA(cudaMemcpy(FILTER_6_INPUT, FILTER_5_OUTPUT, filter5_numberOfResultsPerRun*sizeof(*FILTER_5_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
					#else
						// for (uint64_t i = 0; i < filter5_numberOfResultsPerRun; ++i) {
						// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
						// }
						if (!memcpy(FILTER_6_INPUT, FILTER_5_OUTPUT, filter5_numberOfResultsPerRun*sizeof(*FILTER_5_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of FILTER_5_OUTPUT to FILTER_6_INPUT.\n", filter5_numberOfResultsPerRun*sizeof(*FILTER_5_OUTPUT));
					#endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
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
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Filter 6 caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter6_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter6_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					filter6_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!filter6_numberOfResultsPerRun) continue;


				if (static_cast<void *>(FILTER_7_INPUT) != static_cast<void *>(FILTER_6_OUTPUT)) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(filter6_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_6_OUTPUT, FILTER_7_INPUT);
						// transferResults<<<filter6_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_6_OUTPUT, FILTER_7_INPUT);
						TRY_CUDA(cudaMemcpy(FILTER_7_INPUT, FILTER_6_OUTPUT, filter6_numberOfResultsPerRun*sizeof(*FILTER_6_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
					#else
						// for (uint64_t i = 0; i < filter6_numberOfResultsPerRun; ++i) {
						// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
						// }
						if (!memcpy(FILTER_7_INPUT, FILTER_6_OUTPUT, filter6_numberOfResultsPerRun*sizeof(*FILTER_6_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of FILTER_6_OUTPUT to FILTER_7_INPUT.\n", filter6_numberOfResultsPerRun*sizeof(*FILTER_6_OUTPUT));
					#endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
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
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Filter 7 caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter7_numberOfResultsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter7_numberOfResultsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					filter7_numberOfResultsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!filter7_numberOfResultsPerRun) continue;


				if (static_cast<void *>(FILTER_8_STRUCTURESEED_INPUT) != (currentPopulationChunkDataIndex == 1 ? static_cast<void *>(POPULATION_REVERSAL_OUTPUT) : static_cast<void *>(FILTER_8_OUTPUT))) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_8_STRUCTURESEED_INPUT);
						// transferResults<<<totalStructureSeedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_8_STRUCTURESEED_INPUT);
						TRY_CUDA(cudaMemcpy(FILTER_8_STRUCTURESEED_INPUT, currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(currentPopulationChunkDataIndex == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), cudaMemcpyKind::cudaMemcpyDefault));
					#else
						// for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
						// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
						// }
						if (!memcpy(FILTER_8_STRUCTURESEED_INPUT, currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(currentPopulationChunkDataIndex == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of %s to FILTER_8_STRUCTURESEED_INPUT.\n", totalStructureSeedsPerRun*(currentPopulationChunkDataIndex == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), currentPopulationChunkDataIndex == 1 ? "POPULATION_REVERSAL_OUTPUT" : "FILTER_8_OUTPUT");
					#endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
				}
				if (static_cast<void *>(FILTER_8_TREECHUNK_INPUT) != static_cast<void *>(FILTER_7_OUTPUT)) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(filter7_numberOfResultsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_7_OUTPUT, FILTER_8_TREECHUNK_INPUT);
						// transferResults<<<filter7_numberOfResultsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(FILTER_7_OUTPUT, FILTER_8_TREECHUNK_INPUT);
						TRY_CUDA(cudaMemcpy(FILTER_8_TREECHUNK_INPUT, FILTER_7_OUTPUT, filter7_numberOfResultsPerRun*sizeof(*FILTER_7_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
					#else
						// for (uint64_t i = 0; i < filter7_numberOfResultsPerRun; ++i) {
						// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
						// }
						if (!memcpy(FILTER_8_TREECHUNK_INPUT, FILTER_7_OUTPUT, filter7_numberOfResultsPerRun*sizeof(*FILTER_7_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of FILTER_7_OUTPUT to FILTER_8_TREECHUNK_INPUT.\n", filter7_numberOfResultsPerRun*sizeof(*FILTER_7_OUTPUT));
					#endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
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

			if (static_cast<bool>(ACTUAL_TYPES_TO_OUTPUT & OutputType::Structure_Seeds)) {
				if (static_cast<void *>(PRINT_INPUT) != (ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? static_cast<void *>(POPULATION_REVERSAL_OUTPUT) : static_cast<void *>(FILTER_8_OUTPUT))) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, PRINT_INPUT);
						// transferResults<<<totalStructureSeedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, PRINT_INPUT);
						TRY_CUDA(cudaMemcpy(PRINT_INPUT, ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), cudaMemcpyKind::cudaMemcpyDefault));
					#else
						// for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
						// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
						// }
						if (!memcpy(PRINT_INPUT, ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of %s to PRINT_INPUT.\n", totalStructureSeedsPerRun*(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? "POPULATION_REVERSAL_OUTPUT" : "FILTER_8_OUTPUT");
					#endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
				}

				for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
					// TODO: Possibly replace with toString(OutputType) when implemented?
					if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") {
						std::fprintf(outputFile, "%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), ACTUAL_TYPES_TO_OUTPUT == OutputType::Structure_Seeds ? "" : "\t(Structure seed)");
						std::fflush(outputFile);
					}
					if (!SILENT_MODE) std::printf("%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), ACTUAL_TYPES_TO_OUTPUT == OutputType::Structure_Seeds ? "" : "\t(Structure seed)");
				}
				if (static_cast<int32_t>(ACTUAL_TYPES_TO_OUTPUT) <= 2*static_cast<int32_t>(OutputType::Structure_Seeds) - 1) continue;
			}

			if (static_cast<void *>(WORLDSEED_FILTER_INPUT) != (ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? static_cast<void *>(POPULATION_REVERSAL_OUTPUT) : static_cast<void *>(FILTER_8_OUTPUT)))
			#if CUDA_IS_PRESENT
				// transferResults<<<constexprCeil(static_cast<double>(totalStructureSeedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, WORLDSEED_FILTER_INPUT);
				// transferResults<<<totalStructureSeedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, WORLDSEED_FILTER_INPUT);
				TRY_CUDA(cudaMemcpy(WORLDSEED_FILTER_INPUT, ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), cudaMemcpyKind::cudaMemcpyDefault));
			#else
				// for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) {
				// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
				// }
				if (!memcpy(WORLDSEED_FILTER_INPUT, ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, totalStructureSeedsPerRun*(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of %s to WORLDSEED_FILTER_INPUT.\n", totalStructureSeedsPerRun*(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? sizeof(*POPULATION_REVERSAL_OUTPUT) : sizeof(*FILTER_8_OUTPUT)), ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? "POPULATION_REVERSAL_OUTPUT" : "FILTER_8_OUTPUT");
			#endif
			// TRY_CUDA(cudaGetLastError());
			// TRY_CUDA(cudaDeviceSynchronize());

			for (largeBiomesFlag = static_cast<int32_t>(LARGE_BIOMES_WORLDSEEDS == LargeBiomesFlag::Unknown ? LargeBiomesFlag::False : LARGE_BIOMES_WORLDSEEDS); largeBiomesFlag <= static_cast<int32_t>(LARGE_BIOMES_WORLDSEEDS == LargeBiomesFlag::Unknown ? LargeBiomesFlag::True : LARGE_BIOMES_WORLDSEEDS); ++largeBiomesFlag) {
				totalWorldseedsPerRun = 0;
				#if CUDA_IS_PRESENT
					// biomeFilter<<<constexprMin(constexprCeil(static_cast<double>(totalStructureSeedsPerRun*(static_cast<int32_t>(TYPES_TO_OUTPUT) & (static_cast<int32_t>(ExperimentalOutputType::Random_Worldseeds) | static_cast<int32_t>(OutputType::All_Worldseeds)) ? 65536. : 1.))/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
					biomeFilter<<<constexprMin(totalStructureSeedsPerRun*(static_cast<int32_t>(TYPES_TO_OUTPUT) & (static_cast<int32_t>(ExperimentalOutputType::Random_Worldseeds) | static_cast<int32_t>(ExperimentalOutputType::All_Worldseeds)) ? 65536 : 1)/ACTUAL_WORKERS_PER_BLOCK + 1, static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < totalStructureSeedsPerRun; ++i) pthread_create(&threads[i], NULL, biomeFilter, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (totalWorldseedsPerRun > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Biome filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalWorldseedsPerRun, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalWorldseedsPerRun - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					totalWorldseedsPerRun = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}

				if (static_cast<void *>(PRINT_INPUT) != static_cast<void *>(WORLDSEED_FILTER_OUTPUT)) {
					#if CUDA_IS_PRESENT
						// transferResults<<<constexprCeil(static_cast<double>(totalWorldseedsPerRun)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>(WORLDSEED_FILTER_OUTPUT, PRINT_INPUT);
						// transferResults<<<totalWorldseedsPerRun/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>(WORLDSEED_FILTER_OUTPUT, PRINT_INPUT);
						TRY_CUDA(cudaMemcpy(PRINT_INPUT, WORLDSEED_FILTER_OUTPUT, totalWorldseedsPerRun*sizeof(*WORLDSEED_FILTER_OUTPUT), cudaMemcpyKind::cudaMemcpyDefault));
					#else
						// for (uint64_t i = 0; i < totalWorldseedsPerRun; ++i) {
						// 	pthread_create(&threads[i], NULL, transferResults, &data[i]);
						// }
						if (!memcpy(PRINT_INPUT, WORLDSEED_FILTER_OUTPUT, totalWorldseedsPerRun*sizeof(*WORLDSEED_FILTER_OUTPUT))) ABORT("ERROR: Failed to copy %" PRIu64 " elements of WORLDSEED_FILTER_OUTPUT to PRINT_INPUT.\n", totalWorldseedsPerRun*sizeof(*WORLDSEED_FILTER_OUTPUT));
					#endif
					// TRY_CUDA(cudaGetLastError());
					// TRY_CUDA(cudaDeviceSynchronize());
				}

				for (uint64_t i = 0; i < totalWorldseedsPerRun; ++i) {
					const char *worldseedAttributes = getWorldseedAttributes(PRINT_INPUT[i], largeBiomesFlag);
					if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") {
						std::fprintf(outputFile, "%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), worldseedAttributes);
						std::fflush(outputFile);
					}
					if (!SILENT_MODE) std::printf("%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), worldseedAttributes);
				}
			}
		}
	}

	if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") std::fclose(outputFile);

	#if (!CUDA_IS_PRESENT)
		free(threads);
	#endif
	return 0;
}