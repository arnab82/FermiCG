using QCBase
using ClusterMeanField
using NPZ
using InCoreIntegrals
using RDM
using JLD2
using Printf
using ActiveSpaceSolvers
C = npzread("mo_coeffs.npy")
h0 = npzread("ints_h0.npy")
h1 = npzread("ints_h1.npy")
h2 = npzread("ints_h2.npy")
ints = InCoreInts(h0, h1, h2)

Pa = npzread("Pa.npy")
Pb = npzread("Pb.npy")
@printf(" Input energy:    %12.8f\n", compute_energy(ints, RDM1(Pa, Pb)))


init_fspace=  [(3, 0), (3, 3), (3, 0)]
clusters   =  [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13]]

clusters = [MOCluster(i, collect(clusters[i])) for i = 1:length(clusters)]
display(clusters)

rdm1 = RDM1(n_orb(ints))

ansatze=[FCIAnsatz(5,3,0), FCIAnsatz(3,3,3), FCIAnsatz(5,0,3)]
@time e_cmf, U, d1 = ClusterMeanField.cmf_oo_newton(ints, clusters, init_fspace,ansatze,rdm1, maxiter_oo = 400,
                           tol_oo=1e-8, 
                           tol_d1=1e-9, 
                           tol_ci=1e-11,
                           verbose=4, 
                           zero_intra_rots = false,
                           sequential=true)


d1=orbital_rotation(d1, U)
ints = orbital_rotation(ints, U)
Ccmf=C* U

dip_mo  = npzread("dipole_ints.npy")   # shape [3, n_act, n_act] in Julia
dip_cmf = similar(dip_mo)
for x in 1:3
    dip_cmf[x,:,:] .= U' * dip_mo[x,:,:] * U
end

nabla_mo  = npzread("nabla_ints.npy")
nabla_cmf = similar(nabla_mo)
for x in 1:3
    nabla_cmf[x,:,:] .= U' * nabla_mo[x,:,:] * U
end

@save "data_cmf.jld2" clusters init_fspace ints d1 e_cmf U dip_cmf nabla_cmf Ccmf
