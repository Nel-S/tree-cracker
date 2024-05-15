#ifndef __GENERATE_AND_VALIDATE_DATA_CUH
#define __GENERATE_AND_VALIDATE_DATA_CUH

#include "Experimental Settings.cuh"
#include "Trees Logic.cuh"

__host__ __device__ constexpr const char *getPlural(const int32_t val) noexcept {
	return val == 1 ? "" : "s";
}

// Returns the specfied biome as a string.
__host__ __device__ constexpr const char *toString(const Biome biome) {
	switch (biome) {
		case Biome::Forest: return "Forest";
		case Biome::Birch_Forest: return "Birch Forest";
		case Biome::Taiga: return "Taiga";
		default:
			#if CUDA_IS_PRESENT
				return "";
			#else
				throw std::invalid_argument("Unsupported biome provided.");
			#endif
	}
}

// Returns the specfied version as a string.
__host__ __device__ constexpr const char *toString(const Version version) {
	switch (version) {
		case Version::v1_6_4: return "1.6.4";
		case Version::v1_8_9: return "1.8.9";
		case Version::v1_12_2: return "1.12.2";
		case Version::v1_14_4: return "1.14.4";
		case Version::v1_16_1: return "1.16.1";
		case Version::v1_16_4: return "1.16.4";
		default:
			#if CUDA_IS_PRESENT
				return "";
			#else
				throw std::invalid_argument("Unsupported version provided.");
			#endif
	}
}

// Returns the specified leaf position as a string.
__host__ __device__ constexpr const char *toString(const LeafPosition leafPosition) {
	switch (leafPosition) {
		case LeafPosition::NorthwestLowerLeaf:  return  "Lower Northwest leaf";
		case LeafPosition::SouthwestLowerLeaf:  return  "Lower Southwest leaf";
		case LeafPosition::NortheastLowerLeaf:  return  "Lower Northeast leaf";
		case LeafPosition::SoutheastLowerLeaf:  return  "Lower Southeast leaf";
		case LeafPosition::NorthwestMiddleLeaf: return "Middle Northwest leaf";
		case LeafPosition::SouthwestMiddleLeaf: return "Middle Southwest leaf";
		case LeafPosition::NortheastMiddleLeaf: return "Middle Northeast leaf";
		case LeafPosition::SoutheastMiddleLeaf: return "Middle Southeast leaf";
		case LeafPosition::NorthwestUpperLeaf:  return  "Upper Northwest leaf";
		case LeafPosition::SouthwestUpperLeaf:  return  "Upper Southwest leaf";
		case LeafPosition::NortheastUpperLeaf:  return  "Upper Northeast leaf";
		case LeafPosition::SoutheastUpperLeaf:  return  "Upper Southeast leaf";
		default:
			#if CUDA_IS_PRESENT
				return "";
			#else
				throw std::invalid_argument("Unsupported leaf position provided.");
			#endif
	}
}

// Returns the specified leaf state as a string.
__host__ __device__ constexpr const char *toString(const LeafState leafState) {
	switch (leafState) {
		case LeafState::LeafWasNotPlaced: return "Leaf was not placed";
		case LeafState::LeafWasPlaced: return "Leaf was placed";
		case LeafState::Unknown: return "Unknown";
		default:
			#if CUDA_IS_PRESENT
				return "";
			#else
				throw std::invalid_argument("Unsupported leaf state provided.");
			#endif
	}
}

// Returns the specified leaf state as a string.
__host__ __device__ constexpr const char *toString(const TreeType treetype) {
	switch (treetype) {
		case TreeType::Oak: return "Oak";
		case TreeType::Large_Oak: return "Large Oak";
		case TreeType::Birch: return "Birch";
		case TreeType::Pine: return "Pine";
		case TreeType::Spruce: return "Spruce";
		case TreeType::Unknown: return "Unknown";
		default:
			#if CUDA_IS_PRESENT
				return "";
			#else
				throw std::invalid_argument("Unsupported tree type provided.");
			#endif
	}
}


__host__ __device__ constexpr const char *getNotesAboutSeed(const uint64_t seed, const bool largeBiomesFlag) {
	bool textSeedFlag = (INT32_MIN <= static_cast<int64_t>(seed) && static_cast<int64_t>(seed) <= INT32_MAX);
	bool randomSeedFlag = Random::isFromNextLong(seed);
	// This is constexpr, so I can't use std::string or std::string::c_str().
	// If you have any ideas on a better way to implement this under those constraints, please do.
	if (textSeedFlag) {
		if (randomSeedFlag) {
			if (largeBiomesFlag) return "\t(Text seed, Random seed, Large Biomes seed)";
			return "\t(Text seed, Random seed)";
		}
		if (largeBiomesFlag) return "\t(Text seed, Large Biomes seed)";
		return "\t(Text seed)";
	}
	if (randomSeedFlag) {
		if (largeBiomesFlag) return "\t(Random seed, Large Biomes seed)";
		return "\t(Random seed)";
	}
	if (largeBiomesFlag) return "\t(Large Biomes seed)";
	return "";
}

struct SetOfTreeChunks {
	TreeChunk treeChunks[sizeof(INPUT_DATA)/sizeof(*INPUT_DATA)];
	uint32_t numberOfTreeChunks;

	__device__ constexpr SetOfTreeChunks() noexcept :
		treeChunks(),
		numberOfTreeChunks() {}
	__device__ constexpr SetOfTreeChunks(const InputData data[], const size_t numberOfTreesInData) :
		treeChunks(),
		numberOfTreeChunks(0) {
		// For each tree in the input:
		for (size_t i = 0; i < constexprMax(numberOfTreesInData, sizeof(treeChunks)/sizeof(*treeChunks)); ++i) {
			int32_t treePopulationChunkX = getPopulationChunkCoordinate(data[i].coordinate.x, data[i].version);
			int32_t treePopulationChunkZ = getPopulationChunkCoordinate(data[i].coordinate.z, data[i].version);
			bool matchingChunkExists = false;
			// For each chunk already in the chunks structure:
			for (uint32_t j = 0; j < this->numberOfTreeChunks; ++j) {
				// If a chunk already exists with the specified biome, version, and coordinates, add the tree to it
				if (data[i].biome == this->treeChunks[j].biome && data[i].version == this->treeChunks[j].version && treePopulationChunkX == this->treeChunks[j].populationChunkX && treePopulationChunkZ == this->treeChunks[j].populationChunkZ) {
					this->treeChunks[j].addTree(data[i]);
					matchingChunkExists = true;
					break;
				}
			}
			// Otherwise create a new tree chunk, and add the tree to it
			if (!matchingChunkExists) {
				this->treeChunks[this->numberOfTreeChunks] = TreeChunk(data[i].biome, data[i].version, data[i].coordinate.x, data[i].coordinate.z);
				this->treeChunks[this->numberOfTreeChunks].addTree(data[i]);
				++this->numberOfTreeChunks;
			}
		}

		for (uint32_t i = 0; i < this->numberOfTreeChunks; ++i) {
			// Sort trees inside chunks from most bits to least bits
			for (uint32_t j = 0; j < this->treeChunks[i].numberOfTreePositions; ++j) {
				double currentTreeBits = this->treeChunks[i].treePositions[j].getEstimatedBits(this->treeChunks[i].biome, this->treeChunks[i].version);
				for (uint32_t k = j; k < this->treeChunks[i].numberOfTreePositions; ++k) {
					double comparedTreeBits = this->treeChunks[i].treePositions[k].getEstimatedBits(this->treeChunks[i].biome, this->treeChunks[i].version);
					if (comparedTreeBits > currentTreeBits) {
						TreeChunkPosition temp = this->treeChunks[i].treePositions[j];
						this->treeChunks[i].treePositions[j] = this->treeChunks[i].treePositions[k];
						this->treeChunks[i].treePositions[k] = temp;
						currentTreeBits = comparedTreeBits;
					}
				}
			}

			// Also sort chunks themselves from most total bits to least total bits
			double currentTreeChunkBits = this->treeChunks[i].getEstimatedBits();
			for (uint32_t j = i; j < this->numberOfTreeChunks; ++j) {
				double comparedTreeChunkBits = this->treeChunks[j].getEstimatedBits();
				if (comparedTreeChunkBits > currentTreeChunkBits) {
					TreeChunk temp = this->treeChunks[i];
					this->treeChunks[i] = this->treeChunks[j];
					this->treeChunks[j] = temp;
					currentTreeChunkBits = comparedTreeChunkBits;
				}
			}
		}
	}

	__host__ __device__ constexpr double getEstimatedBits() const noexcept {
		double bits = 0.;
		for (uint32_t i = 0; i < this->numberOfTreeChunks; ++i) bits += this->treeChunks[i].getEstimatedBits();
		return bits - 8.*static_cast<double>(RELATIVE_COORDINATES_MODE); // In Relative Coordinates mode, all trees center around a single position, so if one knows any one position they have all of the other positions immediately. Therefore only 8 bits of information is actually lost.
	}

	void printFindings() const noexcept {
		if (SILENT_MODE) return;
		bool hasPineOrTaigaTrees = false;
		std::fprintf(stderr, "Found %" PRIu32 " population chunk%s. (%.2f bits)\n", this->numberOfTreeChunks, getPlural(this->numberOfTreeChunks), this->getEstimatedBits());
		for (uint32_t i = 0; i < this->numberOfTreeChunks; ++i) {
			// Individual population chunk data
			std::fprintf(stderr, "Chunk #%" PRIu32 ": (%.2f bits)\n", i + 1, this->treeChunks[i].getEstimatedBits());
			std::fprintf(stderr, "\tBiome = %s\n", toString(this->treeChunks[i].biome));
			std::fprintf(stderr, "\tVersion = %s\n", toString(this->treeChunks[i].version));
			std::fprintf(stderr, "\tCoordinate range = (%" PRId32 ", %" PRId32 ") - (%" PRId32 ", %" PRId32 ")\n", getMinBlockCoordinate(this->treeChunks[i].populationChunkX, this->treeChunks[i].version), getMinBlockCoordinate(this->treeChunks[i].populationChunkZ, this->treeChunks[i].version), getMaxBlockCoordinate(this->treeChunks[i].populationChunkX, this->treeChunks[i].version), getMaxBlockCoordinate(this->treeChunks[i].populationChunkZ, this->treeChunks[i].version));
			for (uint32_t j = 0; j < this->treeChunks[i].numberOfTreePositions; ++j) {
				// Individual tree data
				std::fprintf(stderr, "\tTree #%" PRIu32 ": (%.2f bits)\n", j + 1, this->treeChunks[i].treePositions[j].getEstimatedBits(this->treeChunks[i].biome, this->treeChunks[i].version));
				std::fprintf(stderr, "\t\tCoordinate = (%" PRId32 ", %" PRId32 ")\n", getMinBlockCoordinate(this->treeChunks[i].populationChunkX, this->treeChunks[i].version) + static_cast<int32_t>(this->treeChunks[i].treePositions[j].populationChunkXOffset), getMinBlockCoordinate(this->treeChunks[i].populationChunkZ, this->treeChunks[i].version) + static_cast<int32_t>(this->treeChunks[i].treePositions[j].populationChunkZOffset));
				std::fprintf(stderr, "\t\tPossible tree types:\n");
				// Individual tree type data
				if (this->treeChunks[i].treePositions[j].possibleTreeTypes.contains(TreeType::Oak)) {
					// Individual Oak data
					std::fprintf(stderr, "\t\tOak (%.2f bits)\n", this->treeChunks[i].treePositions[j].oakAttributes.getEstimatedBits());
					std::fprintf(stderr, "\t\t\tTrunk height range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].oakAttributes.trunkHeight.lowerBound, this->treeChunks[i].treePositions[j].oakAttributes.trunkHeight.upperBound);
					std::fprintf(stderr, "\t\t\tLeaf states: %" PRIu32 " corner leaves placed, %" PRIu32 " total corner leaf states known\n", getNumberOfOnesIn(this->treeChunks[i].treePositions[j].oakAttributes.leafStates.mask & 0x0000ffff), getNumberOfOnesIn(this->treeChunks[i].treePositions[j].oakAttributes.leafStates.mask & 0xffff0000));
				}
				if (this->treeChunks[i].treePositions[j].possibleTreeTypes.contains(TreeType::Large_Oak)) {
					// Individual Large Oak data
					std::fprintf(stderr, "\t\tLarge Oak (%.2f bits)\n", this->treeChunks[i].treePositions[j].largeOakAttributes.getEstimatedBits());
					std::fprintf(stderr, "\t\t\tTrunk height range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].largeOakAttributes.trunkHeight.lowerBound, this->treeChunks[i].treePositions[j].largeOakAttributes.trunkHeight.upperBound);
				}
				if (this->treeChunks[i].treePositions[j].possibleTreeTypes.contains(TreeType::Birch)) {
					// Individual Birch data
					std::fprintf(stderr, "\t\tBirch (%.2f bits)\n", this->treeChunks[i].treePositions[j].birchAttributes.getEstimatedBits());
					std::fprintf(stderr, "\t\t\tTrunk height range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].birchAttributes.trunkHeight.lowerBound, this->treeChunks[i].treePositions[j].birchAttributes.trunkHeight.upperBound);
					std::fprintf(stderr, "\t\t\tLeaf states: %" PRIu32 " corner leaves placed, %" PRIu32 " total corner leaf states known\n", getNumberOfOnesIn(this->treeChunks[i].treePositions[j].birchAttributes.leafStates.mask & 0x0000ffff), getNumberOfOnesIn(this->treeChunks[i].treePositions[j].birchAttributes.leafStates.mask & 0xffff0000));
				}
				if (this->treeChunks[i].treePositions[j].possibleTreeTypes.contains(TreeType::Pine)) {
					hasPineOrTaigaTrees = true;
					// Individual Pine data
					std::fprintf(stderr, "\t\tPine (%.2f bits)\n", this->treeChunks[i].treePositions[j].pineAttributes.getEstimatedBits());
					std::fprintf(stderr, "\t\t\tTrunk height range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].pineAttributes.trunkHeight.lowerBound, this->treeChunks[i].treePositions[j].pineAttributes.trunkHeight.upperBound);
					std::fprintf(stderr, "\t\t\tLeaves height range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].pineAttributes.leavesHeight.lowerBound, this->treeChunks[i].treePositions[j].pineAttributes.leavesHeight.upperBound);
					std::fprintf(stderr, "\t\t\tLeaves' widest radius range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].pineAttributes.leavesWidestRadius.lowerBound, this->treeChunks[i].treePositions[j].pineAttributes.leavesWidestRadius.upperBound);
				}
				if (this->treeChunks[i].treePositions[j].possibleTreeTypes.contains(TreeType::Spruce)) {
					hasPineOrTaigaTrees = true;
					// Individual Spruce data
					std::fprintf(stderr, "\t\tSpruce (%.2f bits)\n", this->treeChunks[i].treePositions[j].spruceAttributes.getEstimatedBits());
					std::fprintf(stderr, "\t\t\tTrunk height range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].spruceAttributes.trunkHeight.lowerBound, this->treeChunks[i].treePositions[j].spruceAttributes.trunkHeight.upperBound);
					std::fprintf(stderr, "\t\t\tLogs below bottommost leaves' height range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].spruceAttributes.logsBelowBottommostLeaves.lowerBound, this->treeChunks[i].treePositions[j].spruceAttributes.logsBelowBottommostLeaves.upperBound);
					std::fprintf(stderr, "\t\t\tLeaves above trunk's height range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].spruceAttributes.leavesAboveTrunk.lowerBound, this->treeChunks[i].treePositions[j].spruceAttributes.leavesAboveTrunk.upperBound);
					std::fprintf(stderr, "\t\t\tLeaves' widest radius range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].spruceAttributes.leavesWidestRadius.lowerBound, this->treeChunks[i].treePositions[j].spruceAttributes.leavesWidestRadius.upperBound);
					std::fprintf(stderr, "\t\t\tTopmost leaves' radius range: [%" PRId32 ", %" PRId32 "]\n", this->treeChunks[i].treePositions[j].spruceAttributes.topmostLeavesRadius.lowerBound, this->treeChunks[i].treePositions[j].spruceAttributes.topmostLeavesRadius.upperBound);
				}
			}
		}
		std::fprintf(stderr, "\n\n");
		if (hasPineOrTaigaTrees) std::fprintf(stderr, "Note that some attributes for Pine and Taiga trees are rescaled/modified when being converted into the internal format above, so their values may be different.\n\n");
	}
};

__device__ constexpr SetOfTreeChunks POPULATION_CHUNKS_DATA(INPUT_DATA, sizeof(INPUT_DATA)/sizeof(*INPUT_DATA));
// TODO: Finish getting rid of these
__device__ constexpr TreeChunk FIRST_POPULATION_CHUNK = POPULATION_CHUNKS_DATA.treeChunks[0];
constexpr int32_t MAX_CALLS = biomeMaxRandomCalls(FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version);
constexpr int32_t MAX_TREE_COUNT = biomeMaxTreeCount(FIRST_POPULATION_CHUNK.biome, FIRST_POPULATION_CHUNK.version);
constexpr bool COLLAPSE_NEARBY_SEEDS_FLAG = POPULATION_CHUNKS_DATA.treeChunks[0].version <= Version::v1_12_2;
struct Stage2Results {
	uint64_t structureSeedIndex, treechunkSeed;
};

// TODO: Replace CUDA_IS_PRESENT preprocessor directives with ACTUAL_DEVICE
constexpr Device ACTUAL_DEVICE = CUDA_IS_PRESENT ? DEVICE : static_cast<Device>(ExperimenalDevice::CPU);

constexpr uint64_t ACTUAL_NUMBER_OF_PARTIAL_RUNS = NUMBER_OF_PARTIAL_RUNS == static_cast<uint64_t>(AUTO) ? UINT64_C(1) : constexprMax(NUMBER_OF_PARTIAL_RUNS, UINT64_C(1));
constexpr uint64_t ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM = PARTIAL_RUN_TO_BEGIN_FROM == static_cast<uint64_t>(AUTO) ? UINT64_C(1) : constexprMin(constexprMax(PARTIAL_RUN_TO_BEGIN_FROM, UINT64_C(1)), NUMBER_OF_PARTIAL_RUNS);

// constexpr uint64_t ACTUAL_NUMBER_OF_WORKERS = NUMBER_OF_WORKERS;
#if CUDA_IS_PRESENT
constexpr uint64_t ACTUAL_WORKERS_PER_BLOCK = constexprMin(WORKERS_PER_BLOCK, NUMBER_OF_WORKERS);
#endif

/* There are 2^48 possible internal Random states.
   If RELATIVE_COORDINATES_MODE is false, the first four bits of it directly correspond to the first tree's x-offset within the population chunk, so that limits the possibilities to 2^44.*/
constexpr uint64_t TOTAL_NUMBER_OF_STATES_TO_CHECK = UINT64_C(1) << (44 + 4*static_cast<int32_t>(RELATIVE_COORDINATES_MODE));
/* One one hand, the first filter finds all internal states that can generate the tree with the most number of bits of information.
   Therefore for a tree with k bits of information, we can expect there to be around 2^(48 - k) results, or 2^(48 - k)/ACTUAL_NUMBER_OF_PARTIAL_RUNS results per run.
   On the other hand, during each iteration we're only actually analyzing at most NUMBER_OF_WORKERS*(MAX_CALLS + 1)*(1 << MAX_TREE_COUNT) entries.
   Therefore the expected number of results we'll need space for is simply the minimum of those two expression, which is then doubled to provide some leeway just in case.
   (Note: this does not currently take into account the fact that 65536 worldseeds correspond to each ultimate structure seed, since then we'd need to make this 65536x larger
   and I suspect the structure seeds will be filtered enough to render that unnecessary.)*/
constexpr uint64_t RECOMMENDED_RESULTS_PER_RUN = 2*constexprMin((UINT64_C(1) << (48 - static_cast<uint32_t>(POPULATION_CHUNKS_DATA.treeChunks[0].treePositions[0].getEstimatedBits(POPULATION_CHUNKS_DATA.treeChunks[0].biome, POPULATION_CHUNKS_DATA.treeChunks[0].version) - 0.5)))/ACTUAL_NUMBER_OF_PARTIAL_RUNS, NUMBER_OF_WORKERS*(MAX_CALLS + 1)*(1 << MAX_TREE_COUNT));
// NVCC error C2148 places a hard limit of 0x7fffffff bytes per array, while the largest-information array we'll be using is Stage2Results.
constexpr uint64_t MOST_POSSIBLE_RESULTS_PER_RUN = constexprMin(static_cast<uint64_t>(constexprCeil(static_cast<double>(TOTAL_NUMBER_OF_STATES_TO_CHECK)/static_cast<double>(ACTUAL_NUMBER_OF_PARTIAL_RUNS))), 0x7fffffff/sizeof(Stage2Results));
constexpr uint64_t ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN = constexprMin(MAX_NUMBER_OF_RESULTS_PER_RUN ? MAX_NUMBER_OF_RESULTS_PER_RUN : RECOMMENDED_RESULTS_PER_RUN, MOST_POSSIBLE_RESULTS_PER_RUN);


#if (!CUDA_IS_PRESENT)
struct ThreadData {
	uint64_t index, start;
};
#endif
__device__ uint64_t filterInputs[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN];
__managed__ uint64_t filterResults[ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN]; // To prevent new results from accidentally erasing the old inputs mid-filter


void printBoundedSettings() {
	if (SILENT_MODE) return;
	if (DEVICE != ACTUAL_DEVICE) std::fprintf(stderr, "WARNING: Program could not be run on preferred device; defaulting to CPU. (Usually this is because CUDA was not detected.)\n");
	if (PARTIAL_RUN_TO_BEGIN_FROM != ACTUAL_NUMBER_OF_PARTIAL_RUNS) std::fprintf(stderr, "WARNING: NUMBER_OF_PARTIAL_RUNS (%" PRIu64 ") must be at least 1, and so was raised to %" PRIu64 ".\n", NUMBER_OF_PARTIAL_RUNS, ACTUAL_NUMBER_OF_PARTIAL_RUNS);
	if (PARTIAL_RUN_TO_BEGIN_FROM != ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM) std::fprintf(stderr, "WARNING: PARTIAL_RUN_TO_BEGIN_FROM (%" PRIu64 ") must be between 1 and the total number of partial runs (%" PRIu64 ") inclusive, and so was clamped to %" PRIu64 ".\n", PARTIAL_RUN_TO_BEGIN_FROM, NUMBER_OF_PARTIAL_RUNS, ACTUAL_PARTIAL_RUN_TO_BEGIN_FROM);
	if (MAX_NUMBER_OF_RESULTS_PER_RUN != ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) std::fprintf(stderr, "WARNING: MAX_NUMBER_OF_RESULTS_PER_RUN (%" PRIu64 ") cannot be larger than the maximum number of possible results (%" PRIu64 "), and so was truncated to %" PRIu64 ".\n", MAX_NUMBER_OF_RESULTS_PER_RUN, MOST_POSSIBLE_RESULTS_PER_RUN, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
}

#endif