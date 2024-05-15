#include "../src/Settings and Input Data Processing.cuh"

#ifndef __EMULATIONS_H
#define __EMULATIONS_H

constexpr uint64_t WORLDSEED = 123;
constexpr Biome BIOME = Biome::Forest;
constexpr Version VERSION = Version::v1_14_4;
constexpr int32_t POPULATION_CHUNK_X = -2;
constexpr int32_t POPULATION_CHUNK_Z = 6;

int main() {
	Random random;
	if (VERSION <= Version::v1_12_2) {
		random.setSeed(WORLDSEED);
		uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
		uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(random.nextLong()) / 2 * 2 + 1);
		uint64_t populationSeed = static_cast<uint64_t>(POPULATION_CHUNK_X) * a + static_cast<uint64_t>(POPULATION_CHUNK_Z) * b ^ WORLDSEED;
		random.setSeed(populationSeed);
	} else {
		random.setSeed(WORLDSEED);
		uint64_t a = random.nextLong() | 1;
		uint64_t b = random.nextLong() | 1;
		uint64_t populationSeed = static_cast<uint64_t>(POPULATION_CHUNK_X * 16) * a + static_cast<uint64_t>(POPULATION_CHUNK_Z * 16) * b ^ WORLDSEED;
		uint64_t decorationSeed = populationSeed + (VERSION <= Version::v1_14_4 ? 60001 : 80001);
		random.setSeed(decorationSeed);
	}

	CoordInclusiveRange generatedTreePositions[biomeMaxTreeCount(BIOME, VERSION)];
	uint32_t numberOfGeneratedTrees = 0;
	for (uint32_t tree = 0; tree < biomeTreeCount(random, BIOME, VERSION); ++tree) {
		int32_t treeX = getMinBlockCoordinate(POPULATION_CHUNK_X, VERSION) + static_cast<int32_t>(random.nextInt(16));
		int32_t treeZ = getMinBlockCoordinate(POPULATION_CHUNK_Z, VERSION) + static_cast<int32_t>(random.nextInt(16));
		TreeType type = getNextTreeType(random, BIOME, VERSION);
		bool treeAlreadyExists = false;
		for (uint32_t j = 0; j < numberOfGeneratedTrees; ++j) treeAlreadyExists |= generatedTreePositions[j].contains({treeX, treeZ});
		if (treeAlreadyExists) continue;

		uint32_t trunkHeight;
		LeafState leafStates[NUMBER_OF_LEAF_POSITIONS];
		switch (type) {
			case TreeType::Oak:
			case TreeType::Birch:
				trunkHeight = VERSION <= Version::v1_14_4 ? TrunkHeight::getNextValueInRange(random, 4 + (type == TreeType::Birch), 2) : TrunkHeight::getNextValueInRange(random, 4 + (type == TreeType::Birch), 2, 0);
				// if (Version::v1_14_4 < VERSION && VERSION <= Version::v1_16_1) random.skip<2>();
				if (VERSION <= Version::v1_14_4) {
					for (size_t i = 0; i < NUMBER_OF_LEAF_POSITIONS; ++i) leafStates[i] = random.nextBoolean() ? LeafState::LeafWasPlaced : LeafState::LeafWasNotPlaced;
					random.skip<3>();
				} else {
					random.skip<6>();
					for (size_t i = 0; i < NUMBER_OF_LEAF_POSITIONS; ++i) leafStates[8 + i - 8*(i/4)] = random.nextBoolean() ? LeafState::LeafWasPlaced : LeafState::LeafWasNotPlaced;
					// random.skip<4>();
				}
				break;
		}
		generatedTreePositions[numberOfGeneratedTrees] = {{treeX - 2, treeZ - 2}, {treeX + 2, treeZ + 2}};
		++numberOfGeneratedTrees;


		printf("Tree #%" PRIu32 ":\n", numberOfGeneratedTrees);
		printf("\tCoordinate: (%" PRId32 ", %" PRId32 ")", treeX, treeZ);
		printf("\tType: %s\n", toString(type));
		printf("\t\tTrunk Height: %" PRIu32 "\n", trunkHeight);
		switch (type) {
			case TreeType::Oak:
			case TreeType::Birch:
				for (size_t i = 0; i < NUMBER_OF_LEAF_POSITIONS; ++i) {
					printf("\t\t%s: %s\n", toString(static_cast<LeafPosition>(i)), toString(leafStates[i]));
				}
				break;
			case TreeType::Large_Oak:
				break;
			case TreeType::Pine:
				break;
			case TreeType::Spruce:
				break;
		}
	}
	return 0;
}

#endif