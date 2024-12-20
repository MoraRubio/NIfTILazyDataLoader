cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ProstateMRIAnnotator)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/NIfTILazyDataLoader")
set(EXTENSION_CATEGORY "Machine Learning")
set(EXTENSION_CONTRIBUTORS "Alejandro Mora-Rubio ()")
set(EXTENSION_DESCRIPTION "This is an extension with utilities to handle datasets in NIfTI imaging format.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/NIfTILazyDataLoader.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/NIfTILazyDataLoader/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(NIfTILazyDataLoader)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})