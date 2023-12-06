# Memos
EPIC Memo Series

## List of memos
1. [A Roadmap for Efficient Direct Imaging with Large Radio Interferometer Arrays (Astro2020 APC White Paper](PDFs/001_Astro2020_White_Paper.pdf) - White paper submitted to Decadal Survey outlining state of direct imaging and future directions.
2. [Searching for Crab Giant Pulses with EPIC](PDFs/002_Searching_for_CGP.pdf) - First steps toward searching EPIC data for Crab Giant Pulses. Includes some sensitivity considerations and several examples of data artifacts.
3. [Romein Optimization](PDFs/003_Romein_Optimization.pdf) - Modifications to Romein Kernel - Includes a GPU primer that gives and introduction to GPU specific terminologies; Compares Timing of original to modified romein; Commands to perform code profiling
4. [1D Omniscope/EPIC Hybrid](PDFs/004_OmniEPIC.pdf) - An exploration of an Omniscope/EPIC framework. Includes math theory, description of algorithm to fit a grid to an array, and a few examples. Main punchline is that it's not as efficient as hoped.
5. [Cross-Correlator Module / Romein bug fix](PDFs/005_EPIC_xCorr.pdf) - Briefly describes the xCorr module and explains a bug-fix to enable illumination pattern > 1 in romein gridding
6. [FRB Detectability with EPIC](PDFs/006_Sensitivity_Dispersion_curves.pdf) - Describes the projected dispersive delay and pulse broadening for FRBs at LWA frequencies. We also have a sensitivity curve for integrations at EPIC timescales
7. [EPIC Data Assessment](PDFs/007_EPIC_Data_Assessment.pdf) - Describes the issues dealt with while processing EPIC images offline to produce dynamic spectra or light curves for sources
8. [EPIC vs Beam : Data Comparison](PDFs/008_EPICvsBeam_Data_Comparison.pdf) - Describes our understanding of the differences between EPIC and beam-formed data taken simultaneously. Steps to address these is also mentioned.
9. [EPIC Code Optimizations](PDFs/009_EPIC_Code_Optimizations.md) - Describes profiling of GPU code and various optimizations that lead to high bandwidth and full polarization capability.
10. [EPIC Imager Data Management](PDFs/010_EPIC_Imager_Data_Management.md) - Summarizes data rates and storage requirements, and describes schemes for data transfer between imager and post-processing systems.
11. [Float vs Half Precision Accumulation for EPIC Images](PDFs/011_Float_vs_Half_Precision_Accumulation) - Evaluates the uncertainies in pixel values when using half (16-bit) and float (32-bit) precision for image accumulations.
12. [EPIC Imaging & Source Localization Comparisons](PDFs/012_EPIC_Imaging_and_Localization_Comp.pdf) - Performs direct image comparisons between EPIC system and "Off-the-shelf" imagers such as WSClean. Also describes and characterizes outstanding problems within the python EPIC deployment. 
