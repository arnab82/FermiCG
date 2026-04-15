# after running TPSCI:
γ_ab, γ_ba = FermiCG.compute_1rdm_sf(v0a, cluster_ops)

# spin-orbit coupling matrix element with integral H_soc (norb×norb):
SOC = sum(H_soc[p,q] * γ_ab[p,q,r1,r2] for p in 1:norb, q in 1:norb)

# consistency check: for a spin eigenstate with S=3, M_S=3
# γ_ab[p,q,r,r] should be zero (no β electrons to annihilate)
for p in 1:norb, q in 1:norb
    @printf(" γ_ab[%2i,%2i] = %12.6e\n", p, q, γ_ab[p,q,1,1])
end