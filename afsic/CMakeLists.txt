cmake_minimum_required(VERSION 3.15...3.26)

project(afsic LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# When building directly (not via pip) inside a spack environment,
# use the environment's Python and nanobind.
if(DEFINED ENV{SPACK_ENV})
  set(Python_ROOT_DIR "$ENV{SPACK_ENV}/.spack-env/view" CACHE PATH "Spack environment Python prefix")
  set(Python_FIND_REGISTRY NEVER)
  set(Python_FIND_VIRTUALENV FIRST)
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{SPACK_ENV}/.spack-env/view/lib/python3.12/site-packages/nanobind/cmake")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  afsic_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  src/cpp/afsic_ext.cpp
)

add_subdirectory(src/cpp/mail)
target_link_libraries(afsic_ext PRIVATE smtp_mail_sender)

# Install directive for scikit-build-core
install(TARGETS afsic_ext LIBRARY DESTINATION afsic)
