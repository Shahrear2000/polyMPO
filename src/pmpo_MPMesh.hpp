#ifndef POLYMPO_MPM_H
#define POLYMPO_MPM_H

#include "pmpo_utils.hpp"
#include "pmpo_mesh.hpp"
#include "pmpo_materialPoints.hpp"
#include <cstdlib>
#include <iostream>
#include <utility>

namespace polyMPO{

template <MeshFieldIndex>
const MaterialPointSlice meshFieldIndexToMPSlice;
template <> const MaterialPointSlice meshFieldIndexToMPSlice < MeshF_Vel            > = MPF_Vel;
template <> const MaterialPointSlice meshFieldIndexToMPSlice < MeshF_VtxMass        > = MPF_Mass;
template <> const MaterialPointSlice meshFieldIndexToMPSlice < MeshF_ElmMass        > = MPF_Mass;
template <> const MaterialPointSlice meshFieldIndexToMPSlice < MeshF_RotLatLonIncr  > = MPF_Rot_Lat_Lon_Incr;
template <> const MaterialPointSlice meshFieldIndexToMPSlice < MeshF_OnSurfVeloIncr > = MPF_Vel_Incr;
template <> const MaterialPointSlice meshFieldIndexToMPSlice < MeshF_OceanStressCell> = MPF_OceanStress;

template <MaterialPointSlice>
const MeshFieldIndex MPSliceToMeshFieldIndex;
template <> const MeshFieldIndex MPSliceToMeshFieldIndex < MPF_OpenWaterArea  > = MeshF_OpenWaterArea;

#define maxMPsPerElm 8

class MPMesh{

  public:

    Mesh* p_mesh;
    MaterialPoints* p_MPs;

    //For MPI Communication
    int numOwnersTot, numHalosTot;
    std::vector<int> numOwnersOnOtherProcs;
    std::vector<int> numHalosOnOtherProcs;
    std::vector<int> haloOwnerProcs;
    std::vector<std::vector<int>> haloOwnerLocalIDs;
    std::vector<std::vector<int>> ownerOwnerLocalIDs;
    std::vector<std::vector<int>> ownerHaloLocalIDs;

    void startCommunication();
    
    void communicate_and_take_halo_contributions(
        const Kokkos::View<double**>& meshField,
        int nEntities,
        int numEntries,
        int mode,
        int op);
    
    // Original CPU-staging function
    template <typename ViewType>
    void communicate_and_take_halo_contributions1(
        const ViewType& meshField,
        int nEntities,
        int numEntries,
        int mode,
        int op){

      int self;
      MPI_Comm comm = p_MPs->getMPIComm();
      MPI_Comm_rank(comm, &self);

      Kokkos::Timer timer;
      auto reconVals_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), meshField);

      pumipic::RecordTime("SD: GPU-CPU copy-" + std::to_string(self), timer.seconds());

      timer.reset();
      std::vector<std::vector<int>> recvIDVec;
      std::vector<std::vector<double>> recvDataVec;

      pumipic::RecordTime("SD: Recv Vec Allocation-" + std::to_string(self), timer.seconds());
      
      timer.reset();

      communicateFields1(
          reconVals_host,
          nEntities,
          numEntries,
          mode,
          recvIDVec,
          recvDataVec);

      pumipic::RecordTime("SD: IP Comm-" + std::to_string(self), timer.seconds());

      timer.reset();

      int numProcsTot = recvIDVec.size();

      int totalSize = 0;
      std::vector<int> offsets(numProcsTot, 0);

      for(int i = 0; i < numProcsTot; i++){
        offsets[i] = totalSize;
        totalSize += recvIDVec[i].size();
      }

      Kokkos::View<int*> recvIDGPU("recvIDGPU", totalSize);
      auto hostView =
          Kokkos::View<int*, Kokkos::HostSpace>("recvIDCPU", totalSize);

      for(int i = 0; i < numProcsTot; i++){
        std::copy(
            recvIDVec[i].begin(),
            recvIDVec[i].end(),
            hostView.data() + offsets[i]);
      }

      pumipic::RecordTime("SD: Flatten IDs-" + std::to_string(self), timer.seconds());

      timer.reset();

      Kokkos::deep_copy(recvIDGPU, hostView);
      Kokkos::fence();

      pumipic::RecordTime("SD: Copy CPU-GPU-" + std::to_string(self), timer.seconds());

      timer.reset();

      int totalSize_data = 0;
      std::vector<int> offsets_data(numProcsTot, 0);

      for(int i = 0; i < numProcsTot; i++){
        offsets_data[i] = totalSize_data;
        totalSize_data += recvDataVec[i].size();
      }

      Kokkos::View<double*> recvDataGPU("recvDataGPU", totalSize_data);
      auto hostView_data =
          Kokkos::View<double*, Kokkos::HostSpace>("recvDataCPU", totalSize_data);

      for(int i = 0; i < numProcsTot; i++){
        std::copy(
            recvDataVec[i].begin(),
            recvDataVec[i].end(),
            hostView_data.data() + offsets_data[i]);
      }

      pumipic::RecordTime("SD: Flatten Data-" + std::to_string(self), timer.seconds());

      timer.reset();

      Kokkos::deep_copy(recvDataGPU, hostView_data);
      Kokkos::fence();

      assert(totalSize_data == totalSize * numEntries);

      for(int i = 0; i < numProcsTot; i++){
        assert(recvDataVec[i].size() == recvIDVec[i].size() * numEntries);
      }

      pumipic::RecordTime("SD: Copy CPU-GPU2-" + std::to_string(self), timer.seconds());
 
      timer.reset();

      if(op == 0){
        Kokkos::parallel_for(
            "halo contribution add",
            recvIDGPU.size(),
            KOKKOS_LAMBDA(const int i){
              const int vertex = recvIDGPU(i);

              for(int k = 0; k < numEntries; k++){
                Kokkos::atomic_add(
                    &meshField(vertex,k),
                    recvDataGPU(i * numEntries + k));
              }
            });
      }
      else{
        Kokkos::parallel_for(
            "halo contribution assign",
            recvIDGPU.size(),
            KOKKOS_LAMBDA(const int i){
              const int vertex = recvIDGPU(i);

              for(int k = 0; k < numEntries; k++){
                meshField(vertex,k) =
                    recvDataGPU(i * numEntries + k);
              }
            });
      }

      Kokkos::fence();

      pumipic::RecordTime("SD: Contribution" + std::to_string(self), timer.seconds());
    }

    void communicateFields(
        const std::vector<std::vector<double>>& fieldData,
        const int numEntities,
        const int numEntries,
        int mode,
        std::vector<std::vector<int>>& recvIDVec,
        std::vector<std::vector<double>>& recvDataVec);
   
    template <class ViewType> 
    void communicateFields1(
        const ViewType& fieldData,
        const int numEntities,
        const int numEntries,
        int mode,
        std::vector<std::vector<int>>& recvIDVec,
        std::vector<std::vector<double>>& recvDataVec){
    
      int self, numProcsTot;

      MPI_Comm comm = p_MPs->getMPIComm();

      MPI_Comm_rank(comm, &self);
      MPI_Comm_size(comm, &numProcsTot);

      assert(numEntities == numOwnersTot + numHalosTot);

      std::vector<std::vector<double>> sendDataVec(numProcsTot);

      recvIDVec.resize(numProcsTot);
      recvDataVec.resize(numProcsTot);

      for(int i = 0; i < numProcsTot; i++){
        if(i == self) continue;

        int numToSend = 0;
        int numToRecv = 0;

        if(mode == 0){
          numToSend = numOwnersOnOtherProcs[i];
          numToRecv = numHalosOnOtherProcs[i];
        }
        else{
          numToSend = numHalosOnOtherProcs[i];
          numToRecv = numOwnersOnOtherProcs[i];
        }

        if(numToSend > 0){
          sendDataVec[i].reserve(numToSend * numEntries);
        }

        if(numToRecv > 0){
          recvDataVec[i].resize(numToRecv * numEntries);
          recvIDVec[i].resize(numToRecv);
        }
      }

      if(mode == 0){
        for(int iEnt = 0; iEnt < numHalosTot; iEnt++){
          auto ownerProc = haloOwnerProcs[iEnt];

          for(int iDouble = 0; iDouble < numEntries; iDouble++){
            sendDataVec[ownerProc].push_back(
                fieldData(numOwnersTot + iEnt, iDouble));
          }
        }
      }
      else if(mode == 1){
        for(size_t iProc = 0; iProc < ownerOwnerLocalIDs.size(); iProc++){
          for(auto& ownerID : ownerOwnerLocalIDs[iProc]){
            for(int iDouble = 0; iDouble < numEntries; iDouble++){
              sendDataVec[iProc].push_back(
                  fieldData(ownerID, iDouble));
            }
          }
        }
      }

      std::vector<MPI_Request> requests;
      requests.reserve(4 * numProcsTot);

      for(int proc = 0; proc < numProcsTot; proc++){
        if(proc == self) continue;

        if(mode == 0 && numHalosOnOtherProcs[proc]){
          assert(recvIDVec[proc].size() ==
                 static_cast<size_t>(numHalosOnOtherProcs[proc]));

          assert(recvDataVec[proc].size() ==
                 recvIDVec[proc].size() * static_cast<size_t>(numEntries));

          MPI_Request req3;
          MPI_Request req4;

          MPI_Irecv(
              recvIDVec[proc].data(),
              recvIDVec[proc].size(),
              MPI_INT,
              proc,
              1,
              comm,
              &req3);

          MPI_Irecv(
              recvDataVec[proc].data(),
              recvDataVec[proc].size(),
              MPI_DOUBLE,
              proc,
              2,
              comm,
              &req4);

          requests.push_back(req3);
          requests.push_back(req4);
        }

        if(mode == 0 && numOwnersOnOtherProcs[proc]){
          assert(haloOwnerLocalIDs[proc].size() ==
                 static_cast<size_t>(numOwnersOnOtherProcs[proc]));

          assert(sendDataVec[proc].size() ==
                 haloOwnerLocalIDs[proc].size() * static_cast<size_t>(numEntries));

          MPI_Request req1;
          MPI_Request req2;

          MPI_Isend(
              haloOwnerLocalIDs[proc].data(),
              haloOwnerLocalIDs[proc].size(),
              MPI_INT,
              proc,
              1,
              comm,
              &req1);

          MPI_Isend(
              sendDataVec[proc].data(),
              sendDataVec[proc].size(),
              MPI_DOUBLE,
              proc,
              2,
              comm,
              &req2);

          requests.push_back(req1);
          requests.push_back(req2);
        }

        if(mode == 1 && numOwnersOnOtherProcs[proc]){
          MPI_Request req3;
          MPI_Request req4;

          MPI_Irecv(
              recvIDVec[proc].data(),
              recvIDVec[proc].size(),
              MPI_INT,
              proc,
              1,
              comm,
              &req3);

          MPI_Irecv(
              recvDataVec[proc].data(),
              recvDataVec[proc].size(),
              MPI_DOUBLE,
              proc,
              2,
              comm,
              &req4);

          requests.push_back(req3);
          requests.push_back(req4);
        }

        if(mode == 1 && numHalosOnOtherProcs[proc]){
          MPI_Request req1;
          MPI_Request req2;

          MPI_Isend(
              ownerHaloLocalIDs[proc].data(),
              ownerHaloLocalIDs[proc].size(),
              MPI_INT,
              proc,
              1,
              comm,
              &req1);

          MPI_Isend(
              sendDataVec[proc].data(),
              sendDataVec[proc].size(),
              MPI_DOUBLE,
              proc,
              2,
              comm,
              &req2);

          requests.push_back(req1);
          requests.push_back(req2);
        }
      }

      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    

    MPMesh(Mesh* inMesh, MaterialPoints* inMPs):
      p_mesh(inMesh),
      p_MPs(inMPs) {
    };


    ~MPMesh(){
      delete p_mesh;
      delete p_MPs;
    }


    // MP advection and tracking
    void CVTTrackingEdgeCenterBased(Vec2dView dx);
    void CVTTrackingElmCenterBased(const int printVTPIndex = -1);
    void T2LTracking(Vec2dView dx);
    bool push1P();
    void push_ahead();
    void push_swap();
    void push_swap_pos();
    void push();


    // Used before advection to interpolate fields from mesh to MPs
    // And also before reconstruction
    void calcBasis();


    // Reconstruction
    DoubleView assemblyV0();

    template <MeshFieldIndex meshFieldIndex>
    void assemblyVtx0();

    template <MeshFieldIndex meshFieldIndex>
    void assemblyElm0();

    template <MeshFieldIndex meshFieldIndex>
    void assemblyVtx1();

    void reconstruct_coeff_full();
    void invertMatrix(const Kokkos::View<double**>& vtxMatrices, const double& radius, bool vtxCoeffCalc);
    Kokkos::View<double*[vec3d_nEntries][vec4d_nEntries]> precomputedVtxCoeffs_new;
    Kokkos::View<double*[vec3d_nEntries][vec4d_nEntries]> precomputedElmCoeffs_new;
    Kokkos::View<double*> nearAnEdge;
    Kokkos::View<double**> vtxMatrixMass;


    // Not used currently
    std::map<MeshFieldIndex, std::function<void()>> reconstructSlice =
        std::map<MeshFieldIndex, std::function<void()>>();

    template <MaterialPointSlice index>
    DoubleView wtScaAssembly();

    template <MaterialPointSlice index>
    Vec2dView wtVec2Assembly();

    template <MeshFieldIndex meshFieldIndex>
    void assembly(
        int order,
        MeshFieldType type,
        bool basisWeightFlag,
        bool massWeightFlag);

    template<MeshFieldIndex meshFieldIndex>
    void setReconstructSlice(
        int order,
        MeshFieldType type);

    void reconstructSlices();

    void printVTP_mesh(int printVTPIndex);

    void writeMPTrackingVTP(
        int printVTPIndex,
        int numMPs,
        const Vec3dView& history,
        const Vec3dView& resultLeft,
        const Vec3dView& resultRight,
        const Vec3dView& mpTgtPosArray);

    void calculateStrain();
    void calculateStress(const int constitutive_relation);
    void calculateStressDivergence();

    template<MaterialPointSlice MPSlice>
    void mapMPsToCells(){
      //MP field mesh field mapping
      constexpr MeshFieldIndex meshFieldIndex = MPSliceToMeshFieldIndex<MPSlice>;
      auto mpField   = p_MPs->getData<MPSlice>();
      auto meshField = p_mesh->getMeshField<meshFieldIndex>();
      Kokkos::deep_copy(meshField, 0.0); 

      //MP fields
      auto mpArea = p_MPs->getData<MPF_Area>();
      auto mpPos = p_MPs->getData<MPF_Cur_Pos_XYZ>();

      //Mesh fields and finding number of MPs per cell and areaMP per Cell
      auto nElms = p_mesh->getNumElements();
      const auto elmCenter = p_mesh->getMeshField<polyMPO::MeshF_ElmCenterXYZ>();
      double radius = 1.0;
      if(p_mesh->getGeomType() == geom_spherical_surf)
        radius=p_mesh->getSphereRadius();

      Kokkos::View<int*> nMPsPerCell("nMPsPerCell", nElms);
      Kokkos::View<double*> sumAreaMP("sumAreaMP", nElms);
      Kokkos::deep_copy(nMPsPerCell, 0);
      auto calcMPsCell = PS_LAMBDA(const int& elm, const int& mp, const int& mask){
        if(mask){
          Kokkos::atomic_increment(&nMPsPerCell(elm));
          Kokkos::atomic_add(&sumAreaMP(elm), mpArea(mp, 0));
        }
      };
      p_MPs->parallel_for(calcMPsCell, "calcN_MPsCell");

      //TODO put them as inputs
      bool higherOrderRemap = true;

      //Calculate the matrix
      constexpr int numEntriesMatrix=10;
      Kokkos::View<double*[numEntriesMatrix]> lsMatrices("VtxMatrices", p_mesh->getNumElements());
      Kokkos::deep_copy(lsMatrices, 0);

      auto assemble = PS_LAMBDA(const int& elm, const int& mp, const int& mask) {
        if(mask) { //if material point is 'active'/'enabled'
          if(nMPsPerCell(elm) <= 0) return;
          double wp = mpArea(mp,0)/sumAreaMP(elm);
          Kokkos::atomic_add(&lsMatrices(elm,0), wp);
          Kokkos::atomic_add(&lsMatrices(elm,1), wp*(-elmCenter(elm,0)+mpPos(mp,0))/radius);
          Kokkos::atomic_add(&lsMatrices(elm,2), wp*(-elmCenter(elm,1)+mpPos(mp,1))/radius);
          Kokkos::atomic_add(&lsMatrices(elm,3), wp*(-elmCenter(elm,2)+mpPos(mp,2))/radius);
          Kokkos::atomic_add(&lsMatrices(elm,4), wp*(-elmCenter(elm,0)+mpPos(mp,0)) * 
                                                    (-elmCenter(elm,0)+mpPos(mp,0))/(radius*radius));
          Kokkos::atomic_add(&lsMatrices(elm,5), wp*(-elmCenter(elm,0)+mpPos(mp,0)) * (
                                                     -elmCenter(elm,1)+mpPos(mp,1))/(radius*radius));
          Kokkos::atomic_add(&lsMatrices(elm,6), wp*(-elmCenter(elm,0)+mpPos(mp,0)) * 
                                                    (-elmCenter(elm,2)+mpPos(mp,2))/(radius*radius));
          Kokkos::atomic_add(&lsMatrices(elm,7), wp*(-elmCenter(elm,1)+mpPos(mp,1)) * 
                                                    (-elmCenter(elm,1)+mpPos(mp,1))/(radius*radius));
          Kokkos::atomic_add(&lsMatrices(elm,8), wp*(-elmCenter(elm,1)+mpPos(mp,1)) * 
                                                    (-elmCenter(elm,2)+mpPos(mp,2))/(radius*radius));
          Kokkos::atomic_add(&lsMatrices(elm,9), wp*(-elmCenter(elm,2)+mpPos(mp,2)) * 
                                                    (-elmCenter(elm,2)+mpPos(mp,2))/(radius*radius));
        }
      };
      p_MPs->parallel_for(assemble, "assembly");

      invertMatrix(lsMatrices, radius, false);
      auto precomputedElmCoeffs_l = this->precomputedElmCoeffs_new;

      auto calcCellField= PS_LAMBDA(const int& elm, const int& mp, const int& mask) {
        if(mask) { //if material point is 'active'/'enabled
          if(nMPsPerCell(elm) <= 0) return;
          double wp = mpArea(mp,0)/sumAreaMP(elm);
          Vec3d pXYZ = {mpPos(mp,0) - elmCenter(elm,0), mpPos(mp,1) - elmCenter(elm,1), mpPos(mp,2) - elmCenter(elm,2)};
          double psi = wp * (precomputedElmCoeffs_l(elm, 0, 0) + (precomputedElmCoeffs_l(elm, 0, 1) * pXYZ[0] + 
                                                                  precomputedElmCoeffs_l(elm, 0, 2) * pXYZ[1] +
                                                                  precomputedElmCoeffs_l(elm, 0, 3) * pXYZ[2])/radius);
          Kokkos::atomic_add(&meshField(elm, 0), mpField(mp,0)*psi);
        }
      };
      p_MPs->parallel_for(calcCellField, "calcCellField");
    }

    template<MeshFieldIndex mfIndex>
    void mapCellsToMPs(){

      //MP Field
      constexpr MaterialPointSlice mpSlice = meshFieldIndexToMPSlice<mfIndex>;
      auto mpField   = p_MPs->getData<mpSlice>();
      auto mpPos = p_MPs->getData<MPF_Cur_Pos_XYZ>();

      //Mesh Fields
      auto nElms = p_mesh->getNumElements();
      auto meshField = p_mesh->getMeshField<mfIndex>();
      auto elm2ElmConn = p_mesh->getElm2ElmConn();
      auto gnomProjVtx  = p_mesh->getMeshField<MeshF_VtxGnomProj>();
      auto gnomProjCell = p_mesh->getMeshField<MeshF_CellGnomProj>();
      auto gnomProjCellOnCell = p_mesh->getMeshField<MeshF_CellOnCellGnomProj>();
      auto gnomProjElmCenter = p_mesh->getMeshField<MeshF_ElmCenterGnomProj>();
      bool isRotated = p_mesh->getRotatedFlag();

      //Mps Per Cell
      Kokkos::View<int*> nMPsPerCell("nMPsPerCell", nElms);
      Kokkos::deep_copy(nMPsPerCell, 0);
      auto calcMPsCell = PS_LAMBDA(const int& elm, const int& mp, const int& mask){
        if(mask){
          Kokkos::atomic_increment(&nMPsPerCell(elm));
        }
      };
      p_MPs->parallel_for(calcMPsCell, "calcN_MPsCell");

      const int numEntries = mpSliceToNumEntries<mpSlice>();
      for (int comp=0; comp<numEntries; comp++){

        //Calculate mapped Value
        Kokkos::View<double*> centerValues("centerValues", nElms);
        Kokkos::View<double*> grads("grads", 2*nElms);

        Kokkos::parallel_for("calcMatrix", nElms, KOKKOS_LAMBDA(const int elm){   
          if(nMPsPerCell(elm) <= 0) return;
          double lsMatrix[3]={0.0};
          double rsMatrix[2]={0.0};

          int numConnElms = elm2ElmConn(elm, 0);
          for(int i=1; i<=numConnElms; i++){
            int elmID = elm2ElmConn(elm,i)-1;
            if(elmID >= nElms)
              continue;
            lsMatrix[0] += (gnomProjCellOnCell(elm, i-1, 0) - gnomProjCell(elm, 0)) * (gnomProjCellOnCell(elm, i-1, 0) - gnomProjCell(elm, 0)); 
            lsMatrix[1] += (gnomProjCellOnCell(elm, i-1, 0) - gnomProjCell(elm, 0)) * (gnomProjCellOnCell(elm, i-1, 1) - gnomProjCell(elm, 1));
            lsMatrix[2] += (gnomProjCellOnCell(elm, i-1, 1) - gnomProjCell(elm, 1)) * (gnomProjCellOnCell(elm, i-1, 1) - gnomProjCell(elm, 1));
            rsMatrix[0] += (meshField(elmID, comp)- meshField(elm, comp)) * (gnomProjCellOnCell(elm, i-1, 0) - gnomProjCell(elm, 0));
            rsMatrix[1] += (meshField(elmID, comp)- meshField(elm, comp)) * (gnomProjCellOnCell(elm, i-1, 1) - gnomProjCell(elm, 1));
          }
          double det = lsMatrix[0] * lsMatrix[2] - lsMatrix[1] * lsMatrix[1];
          double grad[2] = {0.0, 0.0};
          if (std::abs(det) > 1.0e-12) {
            grad[0] = (1.0 / det) * ( lsMatrix[2] * rsMatrix[0] - lsMatrix[1] * rsMatrix[1]);
            grad[1] = (1.0 / det) * (-lsMatrix[1] * rsMatrix[0] + lsMatrix[0] * rsMatrix[1]);
          }
          double centerVal = meshField(elm, comp) - grad[0] * gnomProjCell(elm, 0) - grad[1] * gnomProjCell(elm, 1);

          grads(2*elm + 0) = grad[0];
          grads(2*elm + 1) = grad[1];
          centerValues(elm) = centerVal;
        });

        auto calcMPValue = PS_LAMBDA(const int& elm, const int& mp, const int& mask){
          if(mask){
            if(nMPsPerCell(elm) <= 0) return;
            Vec3d position3d(mpPos(mp, 0), mpPos(mp, 1), mpPos(mp, 2));
            if(isRotated){
              position3d[0] = -mpPos(mp, 2);
              position3d[2] = mpPos(mp, 0);
            }
            double mpProjX, mpProjY;
            auto gnomProjElmCenter_sub = Kokkos::subview(gnomProjElmCenter, elm, Kokkos::ALL);
            computeGnomonicProjectionAtPoint(position3d, gnomProjElmCenter_sub, mpProjX, mpProjY);

            mpField(mp, comp) = centerValues(elm) + grads(2*elm + 0) * mpProjX + grads(2*elm + 1) * mpProjY;
          }
        };
        p_MPs->parallel_for(calcMPValue, "calcMPValue");
      }
    }

#ifdef CUDA_AWARE_MPI

    // Cached CUDA-aware MPI communication metadata and per-neighbor GPU buffers.
    // Important change from the previous version:
    // MPI is always given the base pointer of a Kokkos allocation, not
    // "base pointer + offset". This avoids Cray MPICH/GTL CUDA IPC problems.
    bool cudaAwareMPICacheValid = true;
    bool cudaAwareMPIDisabled = false;
    bool cudaAwareMPIEnvChecked = false;
    bool cudaAwareMPIForceCPU = false;
    bool cudaAwareMPILogged = false;

    struct CudaAwareMPIFieldCache{
      bool valid = false;
      int cachedNumProcs = -1;

      std::vector<int> sendCounts;
      std::vector<int> recvCounts;

      std::vector<Kokkos::View<int*>> sendEntityGPUPerProc;
      std::vector<Kokkos::View<int*>> recvIDGPUPerProc;

      std::vector<Kokkos::View<double*>> sendDataGPUPerProc;
      std::vector<Kokkos::View<double*>> recvDataGPUPerProc;
    };

    std::map<std::pair<int, int>, CudaAwareMPIFieldCache> cudaAwareMPICaches;

    bool cudaAwareMPIForceDisabled(){
      if(!cudaAwareMPIEnvChecked){
        const char* value = std::getenv("POLYMPO_DISABLE_CUDA_AWARE_MPI");
        cudaAwareMPIForceCPU =
            value != nullptr && value[0] != '\0' && value[0] != '0';
        cudaAwareMPIEnvChecked = true;
      }

      return cudaAwareMPIForceCPU;
    }

    // Fully CUDA-aware MPI version:
    // Field data is sent/received using GPU pointers. Receive IDs are cached
    // once from the fixed halo/owner mapping and are not sent every call.
    //
    // Important:
    // This function caches communication metadata and GPU buffers per
    // (mode, numEntries). If the communication pattern changes, clear
    // cudaAwareMPICaches before the next call.
    template <typename ViewType>
    void communicate_and_take_halo_contributions1_improved(
        const ViewType& meshField,
        int nEntities,
        int numEntries,
        int mode,
        int op){

      int self, numProcsTot;

      MPI_Comm comm = p_MPs->getMPIComm();

      MPI_Comm_rank(comm, &self);
      MPI_Comm_size(comm, &numProcsTot);

      assert(mode == 0 || mode == 1);
      assert(op == 0 || op == 1);
      assert(nEntities == numOwnersTot + numHalosTot);

      if(cudaAwareMPIDisabled || cudaAwareMPIForceDisabled()){
        communicate_and_take_halo_contributions1(
            meshField,
            nEntities,
            numEntries,
            mode,
            op);
        return;
      }

#ifdef POLYMPO_VERBOSE_MPI
      if(self == 0 && !cudaAwareMPILogged){
        std::cout
            << "[CUDA_AWARE_MPI] Using per-proc cached full GPU-aware MPI path in communicate_and_take_halo_contributions1_improved()"
            << "\n";
        cudaAwareMPILogged = true;
      }
#endif

      Kokkos::Timer timer;

      if(!cudaAwareMPICacheValid){
        cudaAwareMPICaches.clear();
        cudaAwareMPICacheValid = true;
      }

      auto& cudaAwareCache =
          cudaAwareMPICaches[std::make_pair(mode, numEntries)];

      const bool needRebuild =
          (!cudaAwareCache.valid) ||
          (cudaAwareCache.cachedNumProcs != numProcsTot);

      if(needRebuild){

        cudaAwareCache.cachedNumProcs = numProcsTot;

        cudaAwareCache.sendCounts.assign(numProcsTot, 0);
        cudaAwareCache.recvCounts.assign(numProcsTot, 0);

        cudaAwareCache.sendEntityGPUPerProc.clear();
        cudaAwareCache.recvIDGPUPerProc.clear();
        cudaAwareCache.sendDataGPUPerProc.clear();
        cudaAwareCache.recvDataGPUPerProc.clear();

        cudaAwareCache.sendEntityGPUPerProc.resize(numProcsTot);
        cudaAwareCache.recvIDGPUPerProc.resize(numProcsTot);
        cudaAwareCache.sendDataGPUPerProc.resize(numProcsTot);
        cudaAwareCache.recvDataGPUPerProc.resize(numProcsTot);

        for(int proc = 0; proc < numProcsTot; proc++){
          if(proc == self) continue;

          if(mode == 0){
            cudaAwareCache.sendCounts[proc] = numOwnersOnOtherProcs[proc];
            cudaAwareCache.recvCounts[proc] = numHalosOnOtherProcs[proc];
          }
          else{
            cudaAwareCache.sendCounts[proc] = numHalosOnOtherProcs[proc];
            cudaAwareCache.recvCounts[proc] = numOwnersOnOtherProcs[proc];
          }
        }

        for(int proc = 0; proc < numProcsTot; proc++){
          if(proc == self) continue;

          const int sendCount = cudaAwareCache.sendCounts[proc];
          const int recvCount = cudaAwareCache.recvCounts[proc];

          if(sendCount > 0){
            cudaAwareCache.sendEntityGPUPerProc[proc] =
                Kokkos::View<int*>(
                    "cudaAwareMPISendEntityGPUPerProc",
                    sendCount);

            cudaAwareCache.sendDataGPUPerProc[proc] =
                Kokkos::View<double*>(
                    "cudaAwareMPISendDataGPUPerProc",
                    sendCount * numEntries);

            auto sendEntityCPU =
                Kokkos::View<int*, Kokkos::HostSpace>(
                    "sendEntityCPU",
                    sendCount);

            if(mode == 0){
              assert(haloOwnerLocalIDs[proc].size() ==
                     static_cast<size_t>(sendCount));

              int localIndex = 0;

              for(int iEnt = 0; iEnt < numHalosTot; iEnt++){
                int ownerProc = haloOwnerProcs[iEnt];

                if(ownerProc != proc) continue;

                assert(localIndex < sendCount);

                sendEntityCPU(localIndex) = numOwnersTot + iEnt;

                localIndex++;
              }

              assert(localIndex == sendCount);
            }
            else{
              assert(ownerOwnerLocalIDs[proc].size() ==
                     static_cast<size_t>(sendCount));

              for(int i = 0; i < sendCount; i++){
                sendEntityCPU(i) = ownerOwnerLocalIDs[proc][i];
              }
            }

            Kokkos::deep_copy(
                cudaAwareCache.sendEntityGPUPerProc[proc],
                sendEntityCPU);
          }

          if(recvCount > 0){
            cudaAwareCache.recvIDGPUPerProc[proc] =
                Kokkos::View<int*>(
                    "cudaAwareMPIRecvIDGPUPerProc",
                    recvCount);

            cudaAwareCache.recvDataGPUPerProc[proc] =
                Kokkos::View<double*>(
                    "cudaAwareMPIRecvDataGPUPerProc",
                    recvCount * numEntries);

            auto recvIDCPU =
                Kokkos::View<int*, Kokkos::HostSpace>(
                    "recvIDCPU",
                    recvCount);

            if(mode == 0){
              assert(ownerOwnerLocalIDs[proc].size() ==
                     static_cast<size_t>(recvCount));

              for(int i = 0; i < recvCount; i++){
                recvIDCPU(i) = ownerOwnerLocalIDs[proc][i];
              }
            }
            else{
              int localIndex = 0;

              for(int iEnt = 0; iEnt < numHalosTot; iEnt++){
                if(haloOwnerProcs[iEnt] != proc) continue;

                assert(localIndex < recvCount);

                recvIDCPU(localIndex) = numOwnersTot + iEnt;

                localIndex++;
              }

              assert(localIndex == recvCount);
            }

            Kokkos::deep_copy(
                cudaAwareCache.recvIDGPUPerProc[proc],
                recvIDCPU);
          }
        }

        Kokkos::fence();

        cudaAwareCache.valid = true;

        pumipic::RecordTime(
            "SD: CUDA-aware MPI Cache Build m" + std::to_string(mode) +
                " e" + std::to_string(numEntries) + "-" + std::to_string(self),
            timer.seconds());

        timer.reset();
      }

      for(int proc = 0; proc < numProcsTot; proc++){
        if(proc == self) continue;
        if(cudaAwareCache.sendCounts[proc] <= 0) continue;

        auto sendEntityGPU = cudaAwareCache.sendEntityGPUPerProc[proc];
        auto sendDataGPU = cudaAwareCache.sendDataGPUPerProc[proc];
        int sendCount = cudaAwareCache.sendCounts[proc];

        Kokkos::parallel_for(
            "pack cached cuda-aware mpi send buffer per proc",
            sendCount,
            KOKKOS_LAMBDA(const int i){
              int entity = sendEntityGPU(i);

              for(int k = 0; k < numEntries; k++){
                sendDataGPU(i * numEntries + k) =
                    meshField(entity, k);
              }
            });
      }

      Kokkos::fence();

      pumipic::RecordTime(
          "SD: CUDA-aware MPI Pack m" + std::to_string(mode) +
              " e" + std::to_string(numEntries) + "-" + std::to_string(self),
          timer.seconds());

      timer.reset();

      Kokkos::Timer mpiTotalTimer;
      std::vector<MPI_Request> requests;
      requests.reserve(2 * numProcsTot);
      int mpiError = MPI_SUCCESS;

      for(int proc = 0; proc < numProcsTot; proc++){
        if(proc == self) continue;

        if(cudaAwareCache.recvCounts[proc] > 0){
          MPI_Request reqData;

          mpiError = MPI_Irecv(
              cudaAwareCache.recvDataGPUPerProc[proc].data(),
              cudaAwareCache.recvCounts[proc] * numEntries,
              MPI_DOUBLE,
              proc,
              2,
              comm,
              &reqData);
          if(mpiError != MPI_SUCCESS) break;
          requests.push_back(reqData);
        }

        if(cudaAwareCache.sendCounts[proc] > 0){
          MPI_Request reqData;

          mpiError = MPI_Isend(
              cudaAwareCache.sendDataGPUPerProc[proc].data(),
              cudaAwareCache.sendCounts[proc] * numEntries,
              MPI_DOUBLE,
              proc,
              2,
              comm,
              &reqData);
          if(mpiError != MPI_SUCCESS) break;
          requests.push_back(reqData);
        }
      }

      pumipic::RecordTime(
          "SD: CUDA-aware MPI Post m" + std::to_string(mode) +
              " e" + std::to_string(numEntries) + "-" + std::to_string(self),
          timer.seconds());

      timer.reset();

      if(mpiError == MPI_SUCCESS && !requests.empty()){
        mpiError = MPI_Waitall(
            static_cast<int>(requests.size()),
            requests.data(),
            MPI_STATUSES_IGNORE);
      }

      pumipic::RecordTime(
          "SD: CUDA-aware MPI Wait m" + std::to_string(mode) +
              " e" + std::to_string(numEntries) + "-" + std::to_string(self),
          timer.seconds());

      if(mpiError != MPI_SUCCESS){
        cudaAwareMPIDisabled = true;

        if(self == 0){
          std::cout
              << "[CUDA_AWARE_MPI] Device-pointer MPI failed."
              << std::endl;
        }

        if(requests.empty()){
          if(self == 0){
            std::cout
                << "[CUDA_AWARE_MPI] Falling back to CPU-staged communication."
                << std::endl;
          }

          communicate_and_take_halo_contributions1(
              meshField,
              nEntities,
              numEntries,
              mode,
              op);
          return;
        }

        if(self == 0){
          std::cout
              << "[CUDA_AWARE_MPI] Failure happened after MPI requests were posted. "
              << "Set POLYMPO_DISABLE_CUDA_AWARE_MPI=1 before running to force the CPU-staged path."
              << std::endl;
        }

        MPI_Abort(comm, mpiError);
        return;
      }

      pumipic::RecordTime(
          "SD: CUDA-aware MPI Comm m" + std::to_string(mode) +
              " e" + std::to_string(numEntries) + "-" + std::to_string(self),
          mpiTotalTimer.seconds());

      timer.reset();

      for(int proc = 0; proc < numProcsTot; proc++){
        if(proc == self) continue;
        if(cudaAwareCache.recvCounts[proc] <= 0) continue;

        auto recvIDGPU = cudaAwareCache.recvIDGPUPerProc[proc];
        auto recvDataGPU = cudaAwareCache.recvDataGPUPerProc[proc];
        int recvCount = cudaAwareCache.recvCounts[proc];

        if(op == 0){
          Kokkos::parallel_for(
              "halo add cached cuda-aware mpi per proc",
              recvCount,
              KOKKOS_LAMBDA(const int i){
                const int vertex = recvIDGPU(i);

                for(int k = 0; k < numEntries; k++){
#ifdef POLYMPO_ASSUME_UNIQUE_HALO_CONTRIBS
                  meshField(vertex, k) +=
                      recvDataGPU(i * numEntries + k);
#else
                  Kokkos::atomic_add(
                      &meshField(vertex, k),
                      recvDataGPU(i * numEntries + k));
#endif
                }
              });
        }
        else{
          Kokkos::parallel_for(
              "halo assign cached cuda-aware mpi per proc",
              recvCount,
              KOKKOS_LAMBDA(const int i){
                const int vertex = recvIDGPU(i);

                for(int k = 0; k < numEntries; k++){
                  meshField(vertex, k) =
                      recvDataGPU(i * numEntries + k);
                }
              });
        }
      }

      Kokkos::fence();

      pumipic::RecordTime(
          "SD: CUDA-aware MPI Contribution m" + std::to_string(mode) +
              " e" + std::to_string(numEntries) + "-" + std::to_string(self),
          timer.seconds());
    }

#else

    // Fallback path:
    // if CUDA_AWARE_MPI is not defined, use the original GPU-CPU staging function.
    template <typename ViewType>
    void communicate_and_take_halo_contributions1_improved(
        const ViewType& meshField,
        int nEntities,
        int numEntries,
        int mode,
        int op){

      communicate_and_take_halo_contributions1(
          meshField,
          nEntities,
          numEntries,
          mode,
          op);
    }

#endif

};

}//namespace polyMPO end

#endif

