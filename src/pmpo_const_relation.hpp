#ifndef POLYMPO_CONSTITUTIVE_RELATION
#define POLYMPO_CONSTITUTIVE_RELATION

#include <stdlib.h>
#include <iostream>

namespace polyMPO{

#define PUNY  1.0e-11
static constexpr double eccentricity = 2.0;
static constexpr double eccentricitySquared = eccentricity * eccentricity;
static constexpr double dampingTimescaleParameter = 0.36;

KOKKOS_INLINE_FUNCTION
void constitutive_evp(const Vec3d& strain, Vec3d& stress, const double& icePressure, double& replacementPressure, 
                      const double& areaMP, const double& dtElastic, const double& dampingTimescale){
  auto strainDivergence = strain[0] + strain[1];
  auto strainTension = strain[0] - strain[1];
  auto strainShearing = 2*strain[2];
  
  auto stress1 = stress[0] + stress[1];
  auto stress2 = stress[0] - stress[1];
  
  auto Delta = sqrt(strainDivergence*strainDivergence + (strainTension*strainTension + strainShearing*strainShearing)/eccentricitySquared);

  auto pressureCoefficient = icePressure / Kokkos::max(Delta, PUNY);
  replacementPressure = pressureCoefficient * Delta;
  pressureCoefficient = (pressureCoefficient * dtElastic) / (2.0 * dampingTimescale);
 
  auto denominator = 1.0 + (0.5 * dtElastic) / dampingTimescale;

  stress1  = (stress1    +  pressureCoefficient                        * (strainDivergence - Delta))  / denominator;
  stress2  = (stress2    + (pressureCoefficient / eccentricitySquared) *  strainTension             ) / denominator;
  stress[2] = (stress[2] + (pressureCoefficient / eccentricitySquared) *  strainShearing * 0.5) / denominator;
 
  stress[0] = 0.5 * (stress1 + stress2);
  stress[1] = 0.5 * (stress1 - stress2);
}

KOKKOS_INLINE_FUNCTION
void constitutive_linear(const Vec3d& strain, Vec3d& stress){
  double lambda = 1.0;  
  stress[0] = lambda * strain[0];
  stress[1] = lambda * strain[1];
  stress[2] = lambda * strain[2];
}

// ---------------- Ice strength (port of seaice_mpm_ice_strength / icepack_ice_strength) ----------------
enum IceStrengthMode {
  ICE_STRENGTH_MPAS_HIBLER = 0,  // no column physics, or no vertical thermo; no solveStress mask
  ICE_STRENGTH_ICEPACK     = 1   // icepack_ice_strength; 0 where solveStress /= 1
};

struct IceStrengthParams {
  int mode, kstrength, krdgPartic, krdgRedist;
  double Pstar, Cstar, Cf, Cp, muRdg, puny;
};

// private constants of icepack_mechred.F90
static constexpr double icepackGstar   = 0.15;
static constexpr double icepackAstar   = 0.05;
static constexpr double icepackMaxraft = 1.0;
static constexpr double icepackHstar   = 25.0;

KOKKOS_INLINE_FUNCTION
double iceStrengthHibler(const double aice, const double vice, const double Pstar, const double Cstar){
  return Pstar*vice*Kokkos::exp(-Cstar*(1.0-aice));
}

template <int NCAT>
KOKKOS_INLINE_FUNCTION
double iceStrengthRothrock(const double aice0, const double* aicen, const double* vicen,
                           const IceStrengthParams& p){
  const double puny = p.puny;
  const double Gstari = 1.0/icepackGstar;
  const double astari = 1.0/icepackAstar;

  double Gsum[NCAT+2];     // Gsum(-1:ncat) -> Gsum[n+1]
  double apartic[NCAT+1];  // apartic(0:ncat)
  double hrmin[NCAT], hrmax[NCAT], hrexp[NCAT], krdg[NCAT];

  for(int n=0; n<NCAT+2; n++) Gsum[n] = 0.0;
  Gsum[1] = (aice0 > puny) ? aice0 : Gsum[0];
  apartic[0] = 0.0;
  for(int n=1; n<=NCAT; n++){
    apartic[n] = 0.0; hrmin[n-1] = 0.0; hrmax[n-1] = 0.0; hrexp[n-1] = 0.0; krdg[n-1] = 1.0;
    Gsum[n+1] = (aicen[n-1] > puny) ? Gsum[n] + aicen[n-1] : Gsum[n];
  }
  if(Gsum[NCAT+1] > 0.0){
    const double work = 1.0/Gsum[NCAT+1];
    for(int n=0; n<=NCAT; n++) Gsum[n+1] = Gsum[n+1]*work;
  }

  if(p.krdgPartic == 0){
    for(int n=0; n<=NCAT; n++){
      const double Gn = Gsum[n+1], Gnm1 = Gsum[n];
      if(Gn < icepackGstar)
        apartic[n] = Gstari*(Gn - Gnm1)*(2.0 - Gstari*(Gnm1 + Gn));
      else if(Gnm1 < icepackGstar)
        apartic[n] = Gstari*(icepackGstar - Gnm1)*(2.0 - Gstari*(icepackGstar + Gnm1));
    }
  } else if(p.krdgPartic == 1){
    const double xtmp = 1.0/(1.0 - Kokkos::exp(-astari));
    Gsum[0] = Kokkos::exp(-Gsum[0]*astari)*xtmp;
    for(int n=0; n<=NCAT; n++){
      Gsum[n+1] = Kokkos::exp(-Gsum[n+1]*astari)*xtmp;
      apartic[n] = Gsum[n] - Gsum[n+1];
    }
  }

  if(p.krdgRedist == 0){
    for(int n=0; n<NCAT; n++) if(aicen[n] > puny){
      const double hi = vicen[n]/aicen[n];
      hrmin[n] = Kokkos::min(2.0*hi, hi + icepackMaxraft);
      hrmax[n] = 2.0*Kokkos::sqrt(icepackHstar*hi);
      hrmax[n] = Kokkos::max(hrmax[n], hrmin[n]+puny);
      krdg[n]  = 0.5*(hrmin[n] + hrmax[n])/hi;
    }
  } else {
    for(int n=0; n<NCAT; n++) if(aicen[n] > puny){
      double hi = vicen[n]/aicen[n];
      hi = Kokkos::max(hi, puny);
      hrmin[n] = Kokkos::min(2.0*hi, hi + icepackMaxraft);
      hrexp[n] = p.muRdg*Kokkos::sqrt(hi);
      krdg[n]  = (hrmin[n] + hrexp[n])/hi;
    }
  }

  double aksum = apartic[0];
  for(int n=1; n<=NCAT; n++) aksum = aksum + apartic[n]*(1.0 - 1.0/krdg[n-1]);

  double strength = 0.0;
  for(int n=0; n<NCAT; n++){
    if(aicen[n] > puny && apartic[n+1] > 0.0){
      const double hi = vicen[n]/aicen[n];
      double h2rdg;
      if(p.krdgRedist == 0)
        h2rdg = (1.0/3.0)*(hrmax[n]*hrmax[n]*hrmax[n] - hrmin[n]*hrmin[n]*hrmin[n])/(hrmax[n] - hrmin[n]);
      else
        h2rdg = hrmin[n]*hrmin[n] + 2.0*hrmin[n]*hrexp[n] + 2.0*hrexp[n]*hrexp[n];
      strength = strength + apartic[n+1]*(-hi*hi + h2rdg/krdg[n]);
    }
  }
  return p.Cf*p.Cp*strength/aksum;
}

template <int NCAT>
KOKKOS_INLINE_FUNCTION
double iceStrengthMP(const int solveStress,
                     const double* aicen, const double* vicen, const IceStrengthParams& p){
  double aice = 0.0, vice = 0.0;
  for(int n=0; n<NCAT; n++){ aice += aicen[n]; vice += vicen[n]; }
  if(p.mode == ICE_STRENGTH_MPAS_HIBLER) return iceStrengthHibler(aice, vice, p.Pstar, p.Cstar);
  if(solveStress != 1) return 0.0;
  const double aice0 = Kokkos::max(1.0 - aice, 0.0);   // open water, as in icepack_aggregate
  if(p.kstrength == 1) return iceStrengthRothrock<NCAT>(aice0, aicen, vicen, p);
  return iceStrengthHibler(aice, vice, p.Pstar, p.Cstar);
}

}
#endif


