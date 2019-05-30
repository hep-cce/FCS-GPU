################################################################################
# Package: CaloGeoHelpers
################################################################################

# Declare the package name:
atlas_subdir( CaloGeoHelpers )

# Declare the package's dependencies:
atlas_depends_on_subdirs( PUBLIC
                          Control/CxxUtils
                          PRIVATE
                          AtlasTest/TestTools )

# Component(s) in the package:
atlas_add_library( CaloGeoHelpers
                   Root/*.cxx
                   PUBLIC_HEADERS CaloGeoHelpers
                   LINK_LIBRARIES CxxUtils
                   PRIVATE_LINK_LIBRARIES TestTools )

atlas_add_test( CaloPhiRange_test
                SOURCES
                test/CaloPhiRange_test.cxx
                LINK_LIBRARIES CxxUtils TestTools CaloGeoHelpers )

