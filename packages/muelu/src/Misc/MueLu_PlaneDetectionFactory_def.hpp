// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_PLANEDETECTIONFACTORY_DEF_HPP
#define MUELU_PLANEDETECTIONFACTORY_DEF_HPP

#include <algorithm>
#include <numeric>

#include <Xpetra_Matrix.hpp>
//#include <Xpetra_MatrixFactory.hpp>

#include "MueLu_PlaneDetectionFactory_decl.hpp"

#include "MueLu_FactoryManager.hpp"
#include "MueLu_Level.hpp"
#include "MueLu_MasterList.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  RCP<const ParameterList> PlaneDetectionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::GetValidParameterList() const {
    RCP<ParameterList> validParamList = rcp(new ParameterList());

    validParamList->set< RCP<const FactoryBase> >("A",               Teuchos::null, "Generating factory of the matrix A");
    validParamList->set< RCP<const FactoryBase> >("Coordinates",     Teuchos::null, "Generating factory for coordinates");

    return validParamList;
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void PlaneDetectionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::DeclareInput(Level& currentLevel) const {
    Input(currentLevel, "A");
  }

  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void PlaneDetectionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(Level& currentLevel) const {
    FactoryMonitor m(*this, "Plane detection", currentLevel);

    RCP<Matrix> A = Get<RCP<Matrix> >(currentLevel, "A");
    const LO dofsPerNode = A->GetFixedBlockSize();
    const LO numNodes    = A->getNodeNumRows() / dofsPerNode;

    // Extract 3D coordinates on all nodes
    if (currentLevel.GetLevelID() == 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(!currentLevel.IsAvailable("Coordinates"), Exceptions::RuntimeError,
          "PlaneDetectionFactory: Build: Coordinates must be supplied.");
    }
    else
      TEUCHOS_TEST_FOR_EXCEPTION(!currentLevel.IsAvailable("Coordinates"), Exceptions::RuntimeError,
          "PlaneDetectionFactory: Build: Coordinates not transferred to coarser levels.");
    constexpr int numDims = 3;
    RCP<CoordinateMultiVector> coordsMV = Get<RCP<CoordinateMultiVector>>(currentLevel, "Coordinates");
    TEUCHOS_TEST_FOR_EXCEPTION(coordsMV->getNumVectors() != numDims, Exceptions::RuntimeError,
        "PlaneDetectionFactory: Build: Three coordinates vectors must be supplied.");
    ArrayRCP<ArrayRCP<const coordinate_type>> coords(numDims);
    for (int dim = 0; dim < numDims; ++dim) {
      coords[dim] = coordsMV->getData(dim);
      TEUCHOS_TEST_FOR_EXCEPTION(coords[dim].is_null(), Exceptions::RuntimeError,
          "PlaneDetectionFactory: Build: Coordinate vector " << dim << " is null.");
    }

    // Find permutation of coordinate indices where the coordinates {x,y,z} are sorted smallest to largest.
    // Assuming {x,y} are exactly the same along the z-direction (extruded mesh), 
    // the z coordinates will be contiguous for each line
    ArrayRCP<LO> indices = arcp<LO>(numNodes);
    LO* indPtr = indices.getRawPtr();
    std::iota(indPtr, indPtr + numNodes, 0);
    sort_indices(indPtr, coords, numNodes, numDims);

    // Find number of planes by counting number of nodes along first line
    LO numPlanes = 0;
    coordinate_type xfirst = coords[0][indices[0]], yfirst = coords[1][indices[0]];
    for (LO node = 1; node < numNodes; ++node) {
      coordinate_type x = coords[0][indices[node]], y = coords[1][indices[node]];
      if (x != xfirst || y != yfirst) {
        numPlanes = node;
        break;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(numPlanes == 0, Exceptions::RuntimeError,
        "PlaneDetectionFactory: Build: Cannot find number of planes via coordinates.");
    TEUCHOS_TEST_FOR_EXCEPTION(numNodes % numPlanes != 0, Exceptions::RuntimeError,
        "PlaneDetectionFactory: Build: Number of nodes is not evenly divisible by number of planes.");
    // const LO numLines = numNodes / numPlanes;
    GetOStream(Runtime1) << "Number of planes from plane detection: " << numPlanes << std::endl;

    // Find plane id and line id for each node
    ArrayRCP<LO> planeIds = arcp<LO>(numNodes), lineIds = arcp<LO>(numNodes);
    for (LO node = 0; node < numNodes; ++node) {
      planeIds[indices[node]] = node % numPlanes;
      lineIds[indices[node]] = node / numPlanes;
    }

    // Store output data on current level
    Set(currentLevel, "CoarseNumZLayers", numPlanes);
    Set(currentLevel, "LineDetection_Layers", planeIds);
    Set(currentLevel, "LineDetection_VertLineIds", lineIds);
  } // Build

  /* Private member function to construct indices based on sorted coordinates */
  template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
  void PlaneDetectionFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::sort_indices(LO* indices,
      const ArrayRCP<const ArrayRCP<const coordinate_type>>& coords, const LO numNodes, const int numDims,
      const int dim) const {

    std::sort(indices, indices + numNodes, [&](LO indi, LO indj) { return coords[dim][indi] < coords[dim][indj]; });

    LO j, i = 0;
    LO indi = indices[i];
    for (j = 1; j < numNodes; ++j) {
      const LO indj = indices[j];
      if (coords[dim][indj] != coords[dim][indi]) {
        // Sort by next dimension if duplicate values exist
        if (j - i > 1 && dim + 1 < numDims)
          sort_indices(indices + i, coords, j - i, numDims, dim + 1);
        i = j;
        indi = indj;
      }
    }

    // Sort by next dimension if duplicate values exist
    if (i != j && dim + 1 < numDims)
      sort_indices(indices + i, coords, j - i, numDims, dim + 1);

  } // sort_indices

} //namespace MueLu

#endif // MUELU_PLANEDETECTIONFACTORY_DEF_HPP
