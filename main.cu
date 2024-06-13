#include "src/Filters.cuh"
#include <chrono>

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
	for (uint64_t partialRun = ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM; partialRun <= ACTUAL_NUMBER_OF_PARTIAL_RUNS; ++partialRun, printFilterResultCounts()) {
		uint64_t runStartSeed = runEndSeed;
		runEndSeed = constexprRound(static_cast<double>(partialRun)*static_cast<double>(TOTAL_NUMBER_OF_STATES_TO_CHECK)/static_cast<double>(ACTUAL_NUMBER_OF_PARTIAL_RUNS));
		if (!SILENT_MODE) {
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
			filter1_numberOfResultsThisWorkerSet = 0;
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

			if (filter1_numberOfResultsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Single Tree Filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter1_numberOfResultsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter1_numberOfResultsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				filter1_numberOfResultsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!filter1_numberOfResultsThisWorkerSet) continue;
			filter1_totalResultsThisRun += filter1_numberOfResultsThisWorkerSet;
			
			// Filter 2 (States with neighboring states able to generate each tree's type and coordinates)
			transferEntries(FILTER_1_OUTPUT, FILTER_2_INPUT, filter1_numberOfResultsThisWorkerSet);
			filter2_numberOfResultsThisWorkerSet = 0;
			#if CUDA_IS_PRESENT
				// filter2<<<constexprCeil(static_cast<double>(filter1_numberOfResultsThisWorkerSet)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
				filter2<<<filter1_numberOfResultsThisWorkerSet/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < filter1_numberOfResultsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, filter2, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (filter2_numberOfResultsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Theoretical-Coords-and-Types filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter2_numberOfResultsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter2_numberOfResultsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				filter2_numberOfResultsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!filter2_numberOfResultsThisWorkerSet) continue;
			filter2_totalResultsThisRun += filter2_numberOfResultsThisWorkerSet;

			// Filter 3 (States with neighboring states able to generate each tree's type, coordinates, and attributes)
			transferEntries(FILTER_2_OUTPUT, FILTER_3_INPUT, filter2_numberOfResultsThisWorkerSet);
			filter3_numberOfResultsThisWorkerSet = 0;
			#if CUDA_IS_PRESENT
				// filter3<<<constexprCeil(static_cast<double>(filter2_numberOfResultsThisWorkerSet)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
				filter3<<<filter2_numberOfResultsThisWorkerSet/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < filter2_numberOfResultsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, filter3, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (filter3_numberOfResultsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Theoretical-Attributes filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter3_numberOfResultsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter3_numberOfResultsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				filter3_numberOfResultsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!filter3_numberOfResultsThisWorkerSet) continue;
			filter3_totalResultsThisRun += filter3_numberOfResultsThisWorkerSet;

			// Treechunk filter (States able to generate the whole chunk for some combination of valid/invalid trees)
			transferEntries(FILTER_3_OUTPUT, TREECHUNK_FILTER_INPUT, filter3_numberOfResultsThisWorkerSet);
			treechunkFilter_numberOfResultsThisWorkerSet = 0;
			#if CUDA_IS_PRESENT
				// treechunkFilter<<<constexprCeil(static_cast<double>(constexprMin(filter3_numberOfResultsThisWorkerSet * (ABSOLUTE_POPULATION_CHUNKS_DATA.getCurrentMaxCalls() + 1) * twoToThePowerOf(ABSOLUTE_POPULATION_CHUNKS_DATA.getCurrentMaxTreeCount()))/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
				treechunkFilter<<<constexprMin(filter3_numberOfResultsThisWorkerSet * (ABSOLUTE_POPULATION_CHUNKS_DATA.getCurrentMaxCalls() + 1) * twoToThePowerOf(ABSOLUTE_POPULATION_CHUNKS_DATA.getCurrentMaxTreeCount())/ACTUAL_WORKERS_PER_BLOCK + 1, static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < filter3_numberOfResultsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, treechunkFilter, &data[i]); // How to handle larger number of i's?
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (treechunkFilter_numberOfResultsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: All-Attributes filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", treechunkFilter_numberOfResultsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, treechunkFilter_numberOfResultsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				treechunkFilter_numberOfResultsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!treechunkFilter_numberOfResultsThisWorkerSet) continue;
			removeDuplicatesAndOrder(TREECHUNK_FILTER_OUTPUT, &treechunkFilter_numberOfResultsThisWorkerSet);
			treechunkFilter_totalResultsThisRun += treechunkFilter_numberOfResultsThisWorkerSet;

			// Print treechunk seeds if desired
			if (static_cast<bool>(ACTUAL_TYPES_TO_OUTPUT & static_cast<OutputType>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds))) {
				transferEntries(TREECHUNK_FILTER_OUTPUT, PRINT_INPUT, treechunkFilter_numberOfResultsThisWorkerSet);
				for (uint64_t i = 0; i < treechunkFilter_numberOfResultsThisWorkerSet; ++i) {
					// TODO: Possibly replace with toString(OutputType) when implemented?
					if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") {
						std::fprintf(outputFile, "%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), ACTUAL_TYPES_TO_OUTPUT == static_cast<OutputType>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds) ? "" : "\t(Highest-information Treechunk seed)");
						std::fflush(outputFile);
					}
					if (!SILENT_MODE) std::printf("%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), ACTUAL_TYPES_TO_OUTPUT == static_cast<OutputType>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds) ? "" : "\t(Highest-information Treechunk seed)");
				}
				// If no structure seeds or worldseeds were flagged to be printed, skip the rest of the filters
				if (static_cast<int32_t>(ACTUAL_TYPES_TO_OUTPUT) <= 2*static_cast<int32_t>(ExperimentalOutputType::Highest_Information_Treechunk_Seeds) - 1) continue;
			}

			// Population chunk reversal (returns potential structure seeds)
			transferEntries(TREECHUNK_FILTER_OUTPUT, POPULATION_REVERSAL_INPUT, treechunkFilter_numberOfResultsThisWorkerSet);
			totalStructureSeedsThisWorkerSet = 0;
			// From Epic10l2
			// 1.6.4:         ?
			// 1.8.9:         ?
			// 1.12.2:        ./main input.txt output.txt 1 1.12 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 10000
			// 1.14.4:        ./main input.txt output.txt 1 1.13 60001 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 0
			// 1.16.1/1.16.4: ./main input.txt output.txt 1 1.13 80001 treechunkXChunkCoordinate treechunkXChunkCoordinate treechunkZChunkCoordinate treechunkZChunkCoordinate 0 0
			#if CUDA_IS_PRESENT
				// reversePopulationSeeds<<<constexprCeil(static_cast<double>(treechunkFilter_numberOfResultsThisWorkerSet)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
				reversePopulationSeeds<<<treechunkFilter_numberOfResultsThisWorkerSet/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
			#else
				for (uint64_t i = 0; i < treechunkFilter_numberOfResultsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, reversePopulationSeeds, &data[i]);
			#endif
			TRY_CUDA(cudaGetLastError());
			TRY_CUDA(cudaDeviceSynchronize());

			if (totalStructureSeedsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
				if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Population seed reverser caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalStructureSeedsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalStructureSeedsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
				totalStructureSeedsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
			}
			if (!totalStructureSeedsThisWorkerSet) continue;

			// For every other chunk in the input data:
			for (currentPopulationChunkDataIndex = 1; currentPopulationChunkDataIndex < ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks; ++currentPopulationChunkDataIndex) {
				// Filter 5 (Not really a filter, just retrieves treechunk seeds for each structure seed)
				// TODO: Probably removable with some work
				transferEntries(currentPopulationChunkDataIndex == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, FILTER_5_INPUT, totalStructureSeedsThisWorkerSet);
				filter5_numberOfResultsThisWorkerSet = 0;
				#if CUDA_IS_PRESENT
					// filter5<<<constexprCeil(static_cast<double>(totalStructureSeedsThisWorkerSet)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					filter5<<<totalStructureSeedsThisWorkerSet/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < totalStructureSeedsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, filter5, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (filter5_numberOfResultsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Structure-to-treechunk Filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter5_numberOfResultsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter5_numberOfResultsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					filter5_numberOfResultsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!filter5_numberOfResultsThisWorkerSet) goto nextRunSeed;

				// Filter 6 (States with neighboring states able to generate each tree's type and coordinates)
				transferEntries(FILTER_5_OUTPUT, FILTER_6_INPUT, filter5_numberOfResultsThisWorkerSet);
				filter6_numberOfResultsThisWorkerSet = 0;
				#if CUDA_IS_PRESENT
					// filter6<<<constexprCeil(static_cast<double>(filter5_numberOfResultsThisWorkerSet)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					filter6<<<filter5_numberOfResultsThisWorkerSet/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < filter5_numberOfResultsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, filter6, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (filter6_numberOfResultsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Filter 6 caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter6_numberOfResultsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter6_numberOfResultsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					filter6_numberOfResultsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!filter6_numberOfResultsThisWorkerSet) goto nextRunSeed;

				// Filter 7 (States with neighboring states able to generate each tree's type, coordinates, and attributes)
				transferEntries(FILTER_6_OUTPUT, FILTER_7_INPUT, filter6_numberOfResultsThisWorkerSet);
				filter7_numberOfResultsThisWorkerSet = 0;
				#if CUDA_IS_PRESENT
					// filter7<<<constexprCeil(static_cast<double>(filter6_numberOfResultsThisWorkerSet)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					filter7<<<filter6_numberOfResultsThisWorkerSet/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < filter6_numberOfResultsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, filter7, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (filter7_numberOfResultsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Filter 7 caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", filter7_numberOfResultsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, filter7_numberOfResultsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					filter7_numberOfResultsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!filter7_numberOfResultsThisWorkerSet) goto nextRunSeed;

				// Filter 8 (Structure seeds that can generate the current neighboring chunk for some combination of valid/invalid trees)
				transferEntries(POPULATION_REVERSAL_OUTPUT, FILTER_8_STRUCTURESEED_INPUT, totalStructureSeedsThisWorkerSet);
				transferEntries(FILTER_7_OUTPUT, FILTER_8_TREECHUNK_INPUT, filter7_numberOfResultsThisWorkerSet);
				totalStructureSeedsThisWorkerSet = 0;
				#if CUDA_IS_PRESENT
					// filter8<<<constexprCeil(static_cast<double>(filter7_numberOfResultsThisWorkerSet)/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK)), ACTUAL_WORKERS_PER_BLOCK>>>();
					filter8<<<filter7_numberOfResultsThisWorkerSet/ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < filter7_numberOfResultsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, filter8, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (totalStructureSeedsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Filter 8 caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalStructureSeedsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalStructureSeedsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					totalStructureSeedsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!totalStructureSeedsThisWorkerSet) goto nextRunSeed;
			}
			filter5_totalResultsThisRun += filter5_numberOfResultsThisWorkerSet;
			filter6_totalResultsThisRun += filter6_numberOfResultsThisWorkerSet;
			filter7_totalResultsThisRun += filter7_numberOfResultsThisWorkerSet;

			// Pass all structure seeds through a set to remove duplicates (and also order them)
			removeDuplicatesAndOrder(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, &totalStructureSeedsThisWorkerSet);
			totalStructureSeedsThisRun += totalStructureSeedsThisWorkerSet;

			// Print structure seeds if desired
			if (static_cast<bool>(ACTUAL_TYPES_TO_OUTPUT & OutputType::Structure_Seeds)) {
				transferEntries(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, PRINT_INPUT, totalStructureSeedsThisWorkerSet);
				for (uint64_t i = 0; i < totalStructureSeedsThisWorkerSet; ++i) {
					// TODO: Possibly replace with toString(OutputType) when implemented?
					if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") {
						std::fprintf(outputFile, "%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), ACTUAL_TYPES_TO_OUTPUT == OutputType::Structure_Seeds ? "" : "\t(Structure seed)");
						std::fflush(outputFile);
					}
					if (!SILENT_MODE) std::printf("%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), ACTUAL_TYPES_TO_OUTPUT == OutputType::Structure_Seeds ? "" : "\t(Structure seed)");
				}
				// If no worldseeds were flagged to be printed, skip the rest of the filters
				if (static_cast<int32_t>(ACTUAL_TYPES_TO_OUTPUT) <= 2*static_cast<int32_t>(OutputType::Structure_Seeds) - 1) continue;
			}

			transferEntries(ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks == 1 ? POPULATION_REVERSAL_OUTPUT : FILTER_8_OUTPUT, WORLDSEED_FILTER_INPUT, totalStructureSeedsThisWorkerSet);

			// TODO: Any way to make this look nicer with operator= overloading?
			for (largeBiomesFlag = static_cast<int32_t>(LARGE_BIOMES_WORLDSEEDS == LargeBiomesFlag::Unknown ? LargeBiomesFlag::False : LARGE_BIOMES_WORLDSEEDS); largeBiomesFlag <= static_cast<int32_t>(LARGE_BIOMES_WORLDSEEDS == LargeBiomesFlag::Unknown ? LargeBiomesFlag::True : LARGE_BIOMES_WORLDSEEDS); ++largeBiomesFlag) {
				totalWorldseedsThisWorkerSet = 0;
				#if CUDA_IS_PRESENT
					// biomeFilter<<<constexprMin(static_cast<uint64_t>(constexprCeil(static_cast<double>(totalStructureSeedsThisWorkerSet*(static_cast<bool>(TYPES_TO_OUTPUT & (ExperimentalOutputType::Random_Worldseeds | ExperimentalOutputType::All_Worldseeds)) ? 65536 : 1))/static_cast<double>(ACTUAL_WORKERS_PER_BLOCK))), static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
					biomeFilter<<<constexprMin(totalStructureSeedsThisWorkerSet*(static_cast<bool>(TYPES_TO_OUTPUT & (ExperimentalOutputType::Random_Worldseeds | ExperimentalOutputType::All_Worldseeds)) ? 65536 : 1)/ACTUAL_WORKERS_PER_BLOCK + 1, static_cast<uint64_t>(INT32_MAX)), ACTUAL_WORKERS_PER_BLOCK>>>();
				#else
					for (uint64_t i = 0; i < totalStructureSeedsThisWorkerSet; ++i) pthread_create(&threads[i], NULL, biomeFilter, &data[i]);
				#endif
				TRY_CUDA(cudaGetLastError());
				TRY_CUDA(cudaDeviceSynchronize());

				if (totalWorldseedsThisWorkerSet > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) {
					if (!SILENT_MODE) std::fprintf(stderr, "WARNING: Biome filter caught more results (%" PRIu64 ") than maximum number allowed (%" PRIu64 "); ignoring last %" PRIu64 " results.\n", totalWorldseedsThisWorkerSet, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN, totalWorldseedsThisWorkerSet - ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
					totalWorldseedsThisWorkerSet = ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN;
				}
				if (!totalStructureSeedsThisWorkerSet) continue;
				removeDuplicatesAndOrder(WORLDSEED_FILTER_OUTPUT, &totalStructureSeedsThisWorkerSet);
				totalWorldseedsThisRun += totalWorldseedsThisWorkerSet;

				// Print desired types of worldseeds
				transferEntries(WORLDSEED_FILTER_OUTPUT, PRINT_INPUT, totalWorldseedsThisWorkerSet);
				for (uint64_t i = 0; i < totalWorldseedsThisWorkerSet; ++i) {
					const char *worldseedAttributes = getWorldseedAttributes(PRINT_INPUT[i], largeBiomesFlag);
					if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") {
						std::fprintf(outputFile, "%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), worldseedAttributes);
						std::fflush(outputFile);
					}
					if (!SILENT_MODE) std::printf("%" PRId64 "%s\n", static_cast<int64_t>(PRINT_INPUT[i]), worldseedAttributes);
				}
			}
			nextRunSeed: continue;
		}
	}

	if (OUTPUT_FILEPATH != NULL && OUTPUT_FILEPATH != "") std::fclose(outputFile);

	#if (!CUDA_IS_PRESENT)
		free(threads);
	#endif
	return 0;
}