# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
#
# Setup the working environment
#

# Bash location
SETUP_LOCATION=${BASH_SOURCE[0]}
# Zsh fallback
if [[ -z ${BASH_SOURCE[0]+x} ]]; then
  SETUP_LOCATION=${(%):-%N}
fi

export FCSSTANDALONE=$(cd "$(dirname "${SETUP_LOCATION}")" && pwd)

echo "FastCaloSim Standalone"
echo "  Installation path: $FCSSTANDALONE"

if [[ ":$PATH:" != *":$FCSSTANDALONE/bin:"* ]];
then
  export PATH="$FCSSTANDALONE/bin:$PATH"
  echo "  Added '\$FCSSTANDALONE/bin' to \$PATH"
fi

if [[ ":$LD_LIBRARY_PATH:" != *":$FCSSTANDALONE/lib:"* ]];
then
  export LD_LIBRARY_PATH="$FCSSTANDALONE/lib:$LD_LIBRARY_PATH"
  echo "  Added '\$FCSSTANDALONE/lib' to \$LD_LIBRARY_PATH"
fi

# if [[ ":$PYTHONPATH:" != *":$FCSSTANDALONE/python:"* ]];
# then
#   export PYTHONPATH="$FCSSTANDALONE/python:$PYTHONPATH"
#   echo "Added '\$FCSSTANDALONE/python' to \$PYTHONPATH"
# fi
# 
# if [[ ":$ROOT_INCLUDE_PATH:" != *":$FCSSTANDALONE/include:"* ]];
# then
#   export ROOT_INCLUDE_PATH="$FCSSTANDALONE/include:$ROOT_INCLUDE_PATH"
#   echo "Added '\$FCSSTANDALONE/include' to \$ROOT_INCLUDE_PATH"
# fi
