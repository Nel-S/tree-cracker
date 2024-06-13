# Tree Cracker[^1]

This is a fork of [Andrew (Gaider10)'s TreeCracker](https://github.com/Gaider10/TreeCracker) that is intended to be streamlined and easier to use. Instead of having to juggle multiple distinct programs and libraries, one now only needs to input their data and settings into [a single file](./Settings%20(MODIFY%20THIS).cuh), then compile and run this program all at once. This fork also adds several extra features to the program, such as prioritizing high-information chunks and determining in advance whether one's input data is likely to be sufficient.

## Purpose
Given details about a set of Minecraft trees (such as their coordinates, types, and attributes), this code is designed to return a list of <!-- worldseeds --> structure seeds[^2] that could <ins>potentially</ins> generate those exact trees.

## Prerequisites and Limitations
At the time of writing, this program only officially supports
- Java Edition.
- Versions <!-- 1.6.4, 1.8.9, 1.12.2, --> 1.14.4, 1.16.1, 1.16.4, or 1.17.1. (Informally, if a specific version isn't supported, setting the relevant data to the next chronological supported version&mdash;e.g. marking 1.16.2 as 1.16.4&mdash;will sometimes still yield results.)
- Oak (normal or fancy) <!--, Spruce, Pine, --> or Birch trees.
- Forest <!--, Birch Forest, or Taiga --> biomes.

The program also uses CUDA, which requires one's device to have an NVIDIA CUDA-capable GPU installed. NVIDIA's CUDA also [does not support MacOS versions OS X 10.14 or beyond](https://developer.nvidia.com/nvidia-cuda-toolkit-developer-tools-mac-hosts). If either of those requirements disqualify your computer, you can instead run the program on a virtual GPU for free (at the time of writing this, and subject to certain runtime limits) through [Google Colab](https://colab.research.google.com).

If using Windows, you will also need some form of C++ compiler installed; however, there are a myriad of environments that provide one ([Microsoft Visual C++](https://learn.microsoft.com/en-us/cpp/build/reference/compiler-options), though that in turn requires [Visual Studio](https://visualstudio.microsoft.com); [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl); [Minimialist GNU for Windows-w64](https://www.mingw-w64.org); and others).

## Installation, Setup, and Usage
1. Download the repository, either as a ZIP file from GitHub or by cloning it through Git.
2. Open [the Settings file](./Settings%20(MODIFY%20THIS).cuh) in your favorite code editor, and replace the examples of input data with your own, and the settings with your own. (For enumerations like `Version` or `Biome`, the list of supported values can be found in [Allowed Values for Settings.cuh](./Allowed%20Values%20for%20Settings.cuh).)

<!-- TODO: Rework warnings to apply these (i.e. warn about low first-tree and first-treechunk bits instead of total bits) -->
When creating your input data, keep in mind that <ins>the comprehensiveness of the data</ins> (specifically the number of "bits" of information the highest-information tree and treechunk <!-- TODO: Explain treechunks? --> reveal) <ins>matters far more than factors like the number of trees.</ins> For example, I and Chaos4669 once tried to crack the same worldseed using this tool:
 - my input data had eleven trees holding 123.87 combined bits of information, but my highest-information treechunk among that revealed only 54.71 bits and my highest-information tree only 13.06 bits&mdash;which caused the program to return dozens of possibilities and require multiple days before it could finish.
 - Chaos' input data ([`TEST_DATA_16_1_2`](./Test%20Data/1.16.1.cuh)), meanwhile, had half the number of trees and total bits of information, but his highest-information treechunk held 68.93 bits of information and his highest-information tree revealed 19.91 bits of information. With those, the program finished within five minutes and returned a mere four possibilities.

(The program will print the number of bits each piece of input data reveals when the program first begins running. If a multi-day runtime cannot be avoided, the program also comes with the ability to divide one's runs into "partial runs" so that one's run can be resumed midway-through at a later time.)

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
If the output is an error and not the compiler's information, you will need to install the CUDA Toolkit which contains `nvcc`. (The following are installation guides for [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux), [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows), and [MacOS X 10.13 or earlier](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-mac-os-x/).)

6. Navigate in the terminal to the folder where the program's files are contained:
```bash
(Linux/Windows/MacOS)  cd "[Path to the folder]"
(Google Colab)        !cd "[Path to the folder]"
```
Then use `nvcc` to compile the program:
```bash
(Linux)         nvcc main.cu -o "main" -O3
(Windows)       nvcc main.cu -o "main.exe" -O3
(MacOS)         nvcc main.cu -o "main.app" -O3
(Google Colab) !nvcc main.cu -o "main" -O3
```
Depending on your input data, the compilation may take almost a full minute or even longer.<br />
The compiler may print warnings akin to `Stack size for entry function '_Z11biomeFilterv' cannot be statically determined`: this is normal. (All this means is that the compiler couldn't determine the exact number of iterations certain recursive functions will undergo.)

7. Run the compiled program:
```bash
(Linux)         .\main
(Windows)       .\main.exe
(MacOS)         open -a main.app
(Google Colab) !.\main
```
As mentioned in step 2, the program's runtime can vary wildly based on one's input data and its comprehensiveness. Nevertheless, if all goes well, a file should ultimately be created (or a list should be printed to the screen, depending on your settings) containing your possible <!-- worldseeds --> structure seeds.

8. At some point, this program will also automatically filter structure seeds into potential worldseeds. This hasn't been implemented yet, though, so in the meantime one must perform this filtering manually.
    1. Download and open [Cubiomes Viewer](https://github.com/Cubitect/cubiomes-viewer/releases).
    2. Under the Edit tab in the upper top-left, click "Advanced World Settings" and make sure "Enable experimentally supported versions" is enabled.
    3. Close the World Settings menu and set the "MC" input box in the top-left corner to your world's version (or the supported version closest to it).
    4. Under the Seed Generator heading, Click "Seed list", then use the button across from the "Load 48-bit seed list" option to select whichever file contains this program's outputted structure seeds.
    5. For each population chunk in your input data (these will have been displayed when <ins>this</ins> program first began running):
        - Under the Conditions heading, click "Add".
        - Select "Biomes" for the condition's category and "Overworld at scale" as the condition's type.
        - Select Custom for the location and enter the population chunk's coordinate range.
        - Select "1:1 ..." for the Scale/Generation Layer, then exclude all biomes except the population chunk's biome.
    6. When finished adding all conditions, click "Start search" at the bottom of the window. The program will then start outputting worldseeds that have biomes matching your input data.

WARNING: When checking the outputted worldseeds, some generated trees may not match your input data. (Tree generation depends on the order that chunks are loaded, so if the chunks are loaded in a different order than your input data's source, a different pattern of trees will form.) However, in most cases at least a few trees will match your input data; if *every* tree is different, that is an indication your original input data (or this tool) are likely wrong.

## Acknowledgements
I would like to give very large Thank You's to
- [Andrew](https://github.com/Gaider10), for creating the [original version of the TreeCracker](https://github.com/Gaider10/TreeCracker) (alongside much of the test data) and a [population chunk reverser](https://github.com/Gaider10/PopulationCrr), and for answering a question about his tool.
- [Cortex](https://github.com/mcrcortex) and [TatertotGr8](https://github.com/tatertotgr8), for their tree crackers that predate even Andrew's ([Cortex's](https://github.com/MCRcortex/TreeCracker), [TatertotGr8's](https://github.com/TatertotGr8/Treecracker)).
- [Epic10l2](https://github.com/epic10l2), for his [comprehensive guide to Andrew's TreeCracker](https://docs.google.com/document/d/1csrcO2F4qQ2ahYgcicWmJtnfeU99q65p) that enabled me to learn how to use the program.
- [Edd](https://github.com/humhue), for helping considerably with Epic10l2's guide, and for answering a few questions about 1.12.2- population reversal.
- [Neil](https://github.com/hube12), for finding and listing most tree salts ([1.13](https://gist.github.com/hube12/574512a3c4df2be8ba6c08e7298caedd), [1.14](https://gist.github.com/hube12/394ddf11b3cdcc9504270777565446e4), [1.15](https://gist.github.com/hube12/821b66615a97a7130ef804603d68bec8), [1.16](https://gist.github.com/hube12/b65500cd234ce2a3983b62b3903c183d), [1.17](https://gist.github.com/hube12/5066fbcd8565648dd68113a9b065514b)).
- [Chaos4669](https://youtube.com/@Chaotic4669), for providing test data and recommendations on what could benefit from clarification.
- [Cubitect](https://github.com/cubitect), for his [Cubiomes library](https://github.com/Cubitect/cubiomes) that this program (will ultimately) use a port of to filter biomes, and his [Cubiomes Viewer](https://github.com/Cubitect/cubiomes-viewer) GUI tool I recommend as a substitute in the meantime.
- [Panda4994](https://github.com/panda4994), for [his algorithm]((https://github.com/Panda4994/panda4994.github.io/blob/48526d35d3d38750102b9f360dff45a4bdbc50bd/seedinfo/js/Random.js#L16)) to determine if a state is derivable from a nextLong call.

If you would like to contribute to this repository or report any bugs, please feel free to open an issue or a pull request.

This repository is offered under [my (NelS') general seedfinding license](https://github.com/Nel-S/seedfinding/blob/main/LICENSE). Please read and abide by that text if you have any wishes of referencing, distributing, selling, etc. this repository or its code.[^3]

[^1]: ...or more accurately "Tree Brute-forcer", but the previous term is so ingrained in seedcracking culture that I can't exactly change it now.
[^2]: If one converts a worldseed into a 64-bit binary integer, a structure seed corresponds to the worldseed's last 48 bits. Therefore each structure seed has 2<sup>16</sup> = 65536 worldseeds associated with it. ...eventually, biome filtering will be used to directly return worldseeds instead of structure seeds, but this has not been finished yet.
[^3]: While the license discusses this, I want to emphasize one aspect of it here: this repository relies upon numerous others' repositories (Gaider10's TreeCracker, Gaider10's Population Chunk Reverser, Cubitect's Cubiomes library, etc.), and thus my license solely applies to the changes I and any voluntary contributors made within this repository, not to their repositories or any code in this repository that is untouched from their repositories.