# Tree Cracker[^1]

This is a fork of [Andrew (Gaider10)'s TreeCracker](https://github.com/Gaider10/TreeCracker) that is intended to be streamlined and easier to use. Instead of having to juggle multiple distinct programs and libraries, one now only needs to input their data and settings into [a single file](./Settings%20(MODIFY%20THIS).cuh), then compile and run this program all at once. This fork also adds several extra features to the program, such as prioritizing high-information chunks and determining in advance whether one's input data will ultimately be sufficient.

## Purpose
Given details about a set of Minecraft trees (such as their coordinates, types, and attributes), this code is designed to return a list of worldseeds that could <ins>potentially</ins> generate those exact trees.

## Prerequisites and Limitations
At the time of writing, this program only supports
- Java Edition.
- Versions 1.6.4, 1.8.9, 1.12.2, 1.14.4, 1.16.1, or 1.16.4. (Informally, if a specific version isn't supported, setting the relevant data to the next chronological supported version&mdash;e.g. marking 1.7 as 1.8.9&mdash;will sometimes still yield results.)
- Oak (normal or large), Spruce, Pine, or Birch trees.
- Forest, Birch Forest, or Taiga biomes.
If you would like to add support for another version, tree type, or biome, or otherwise improve this code or report a bug with it, please feel free to open a pull request.

The program also uses CUDA, which requires one's device to have an NVIDIA CUDA-capable GPU installed. NVIDIA's CUDA also [does not support MacOS versions OS X 10.14 or beyond](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts). If either of those requirements disqualify your computer, you can instead run the program on a virtual GPU for free (at the time of writing this, and subject to certain runtime limits) through [Google Colab](https://colab.research.google.com).

## Installation, Setup, and Usage
1. Download the repository, either as a ZIP file from GitHub or by cloning it through Git.
2. Open [the Settings file](./Settings%20(MODIFY%20THIS).cuh) in your favorite code editor, and replace the examples of input data with your own, and the settings with your own. (For enumerations like `Version` or `Biome`, the list of supported values can be found in [AllowedValuesForSettings.cuh](./AllowedValuesForSettings.cuh).)
3. Go back and double-check your input data. There is an 80% chance you inputted something incorrectly the first time, and any mistakes will prevent the program from deriving the correct worldseeds.
4. Once you're *completely certain* your input data is correct&mdash;if you wish to run the program on Google Colab:
    1. Visit [the website](https://colab.research.google.com), sign in with a Google account, and create a new notebook.
    2. Open the Files sidebar to the left and upload the program's files, making sure to keep the files' structure the way it originally was (the underlying code files are inside a folder named src, etc.). Don't forget to upload the modified Settings file instead of the original.
    3. Under the Runtime tab, select "Change runtime type" and select T4 GPU as the hardware accelerator.
5. Whether on Google Colab or your own computer, open a terminal and verify [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html), the CUDA compiler, is installed:
```bash
(Linux/Windows/MacOS)  nvcc --version
(Google Colab)        !nvcc --version
```
If the output is an error and not the compiler's information, you will need to install the CUDA Toolkit which contains `nvcc`. (The following are installation guides for [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux), [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows), and [MacOS X 10.13 or before](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-mac-os-x/).)
6. Navigate in the terminal to the folder where the program's files are contained:
```bash
(Linux/Windows/MacOS)  cd "[Path to the folder]"
(Google Colab)        !cd "[Path to the folder]"
```
Then use `nvcc` to compile the program:
```bash
(Linux)         nvcc main.cu -o "main" -O3
(Windows)       nvcc main.cu -o "main.exe" -O3
(Google Colab) !nvcc main.cu -o "main" -O3
```
Depending on your input data, the compilation may take almost a full minute or even longer.
The compilation may print a warning akin to `Stack size for entry function '_Z11biomeFilterv' cannot be statically determined`: do not worry if this happens. (Cubiomes, which this program uses a port of to filter biomes, generates internal "layers" for Beta 1.8&ndash;1.17 recursively, which means the compiler cannot statically calculate in advance exactly how many layers there are.)
7. Run the compiled program:
```bash
(Linux)         .\main
(Windows)       .\main.exe
(MacOS)         open -a main.app
(Google Colab) !.\main
```
After a (potentially significant) amount of time, if all goes well, a file should ultimately be created (or a list should be printed to the screen, depending on your settings) containing your possible worldseeds.

## Acknowledgements
I would like to give a very large Thank You to
- Andrew, for creating the [original version of the TreeCracker](https://github.com/Gaider10/TreeCracker) and a [population chunk reverser](https://github.com/Gaider10/PopulationCrr).
- Cortex and TatertotGr8, for their tree crackers that predate even Andrew's ([Cortex's](https://github.com/MCRcortex/TreeCracker), [TatertotGr8's](https://github.com/TatertotGr8/Treecracker))
- Epic10l2, for his [comprehensive guide to Andrew's TreeCracker](https://docs.google.com/document/d/1csrcO2F4qQ2ahYgcicWmJtnfeU99q65p) that enabled me to learn how to use the program.
- Cubitect, for his [Cubiomes library](https://github.com/Cubitect/cubiomes) that this program uses a port of to filter biomes.
- Panda4994, for [his algorithm]((https://github.com/Panda4994/panda4994.github.io/blob/48526d35d3d38750102b9f360dff45a4bdbc50bd/seedinfo/js/Random.js#L16)) to determine if a state is derivable from a nextLong call.

This repository is offered under the MIT License. However, if you do use or reference this respoitory, I would greatly appreciate a citation.

[^1]: ...or more accurately "Tree Brute-forcer", but the previous term is so ingrained in seedcracking culture that I can't exactly change it now.