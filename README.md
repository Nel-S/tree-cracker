# Tree Cracker[^1]

This is a fork of [Andrew (Gaider10)'s TreeCracker](https://github.com/Gaider10/TreeCracker) that is intended to be simpler and more intuitive to use. Instead of having to juggle various separate programs and external libraries, one now only needs to input their data and settings into [a single file](./Settings%20(MODIFY%20THIS).cuh), then compile and run this program all at once. This fork also adds several extra features to the program, such as the ability to determine in advance whether one's input data will likely be sufficient.

## Purpose and Instructions
Given details about a set of Minecraft trees (such as their coordinates, types, and attributes), this code is designed to return a list of worldseeds that could potentially generate those exact trees.

To install the program:
1. Download the repository, either as a ZIP file or by cloning it through Git.
2. Open [the Settings file](./Settings%20(MODIFY%20THIS).cuh) in one's favorite code editor, and replace the examples of input data with your own, and the settings with your own. (For enumerations like `Version` or `Biome`, the list of supported values can be found in [./AllowedValuesForSettings.cuh].)
3. If you do not have a CUDA GPU on your own computer, you can use one for free (subject to certain runtime limits) on [Google Colab](https://colab.research.google.com).
    1. Visit the website, log in with your Google account, and create a new notebook.
    2. Open the Files sidebar and upload the program's files, including the modified Settings file.
    3. Under the Runtime tab, select "Change runtime type" and select T4 GPU as the hardware accelerator.
Otherwise if you do have a CUDA GPU on your own computer, make sure [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) is installed.
4. In a terminal, navigate to the folder where the program's files are contained:
```bash
(Linux/Windows)  cd "[Path to the folder]"
(Google Colab)  !cd "[Path to the folder]"
```
Then use nvcc to compile the program:
```bash
(Linux)         nvcc main.cu -o "main" -O3
(Windows)       nvcc main.cu -o "main.exe" -O3
(Google Colab) !nvcc main.cu -o "main" -O3
```
5. Run the compiled program:
```bash
(Linux)         .\main
(Windows)       .\main.exe
(Google Colab) !.\main
```

If all goes well, a file should be created (or a list should be printed to the screen, depending on one's settings) containing a list of possible worldseeds.

## Limitations
At the time of writing, this program only supports
- Versions 1.6.4, 1.8.9, 1.12.2, 1.14.4, 1.16.1, or 1.16.4.
- Oak (normal or large), Spruce, Pine, or Birch trees.
- Forest, Birch Forest, or Taiga biomes.
If you would like to add support for another version, tree type, or biome, or otherwise improve this code or report a bug with it, please feel free to open a pull request.

This repository is offered under the MIT License. However, if you do use or reference this respoitory, I would greatly appreciate a citation.

## Acknowledgements
I would like to give a very large Thank You to
- Andrew, for creating the [original version of the TreeCracker](https://github.com/Gaider10/TreeCracker) and a [population chunk reverser](https://github.com/Gaider10/PopulationCrr).
- Cortex and TatertotGr8, for their tree crackers that predate even Andrew's ([Cortex's](https://github.com/MCRcortex/TreeCracker), [TatertotGr8's](https://github.com/TatertotGr8/Treecracker))
- Epic10l2, for his [comprehensive guide to Andrew's TreeCracker](https://docs.google.com/document/d/1csrcO2F4qQ2ahYgcicWmJtnfeU99q65p) that enabled me to learn how to use the program.
- Cubitect, for his [Cubiomes library](https://github.com/Cubitect/cubiomes) that this program uses to filter biomes.
- Panda4994, for [his algorithm]((https://github.com/Panda4994/panda4994.github.io/blob/48526d35d3d38750102b9f360dff45a4bdbc50bd/seedinfo/js/Random.js#L16)) to determine if a state could be derived from a nextLong call.

[^1]: ...or more accurately a tree brute-forcer, but the term is so ingrained in seedfinding circles that I can't just change it now.