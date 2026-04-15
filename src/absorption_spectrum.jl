"""
absorption_spectrum.jl

Oscillator strengths and UV/Vis absorption spectra from one-electron
dipole integrals and TPSCI transition 1-RDMs.

Typical workflow:
    # 1. Solve TPSCI for R roots
    E, psi = tpsci_ci(...)

    # 2. Compute all inter-root transition 1-RDMs at once
    γ_aa, γ_bb = compute_1rdm(psi, cluster_ops)

    # 3. Get oscillator strengths (dipole_x/y/z are norb×norb matrices)
    f = compute_oscillator_strengths(E, γ_aa, γ_bb, dipole_x, dipole_y, dipole_z)

    # 4. Build the spectrum
    ω, I = absorption_spectrum(E, f)
"""


"""
    compute_transition_dipoles(γ_aa, γ_bb, dip_x, dip_y, dip_z; ref_root=1)

Compute transition dipole moments from the reference root to all other roots.

Returns (tdm_x, tdm_y, tdm_z), each a Vector of length R,
where tdm_α[n] = Σ_{pq} dip_α[p,q] * (γ_aa[p,q,ref_root,n] + γ_bb[p,q,ref_root,n]).

The diagonal element (n == ref_root) is the permanent dipole of that root.
"""
function compute_transition_dipoles(γ_aa::Array{T,4}, γ_bb::Array{T,4},
                                     dip_x::AbstractMatrix{T},
                                     dip_y::AbstractMatrix{T},
                                     dip_z::AbstractMatrix{T};
                                     ref_root::Int=1) where T
    _, _, R1, R2 = size(γ_aa)
    R1 == R2 || throw(ArgumentError("γ must be square in root indices (R1==R2) for single-state use"))
    R = R1

    γ_total = γ_aa .+ γ_bb   # (norb, norb, R, R)

    tdm_x = zeros(T, R)
    tdm_y = zeros(T, R)
    tdm_z = zeros(T, R)

    for n in 1:R
        @views γ_0n = γ_total[:, :, ref_root, n]
        tdm_x[n] = dot(vec(dip_x), vec(γ_0n))
        tdm_y[n] = dot(vec(dip_y), vec(γ_0n))
        tdm_z[n] = dot(vec(dip_z), vec(γ_0n))
    end

    return tdm_x, tdm_y, tdm_z
end


"""
    compute_oscillator_strengths(E, γ_aa, γ_bb, dip_x, dip_y, dip_z; ref_root=1)

Compute electric-dipole oscillator strengths in the length gauge:

    f_n = (2/3) * (E_n - E_ref) * (|<ref|μ_x|n>|² + |<ref|μ_y|n>|² + |<ref|μ_z|n>|²)

Arguments
---------
- `E`          : Vector of R eigenenergies (in Hartree)
- `γ_aa`, `γ_bb` : transition 1-RDMs from `compute_1rdm(psi, cluster_ops)`,
                   shape (norb, norb, R, R)
- `dip_x/y/z`  : norb×norb one-electron dipole integral matrices
- `ref_root`   : root index of the reference state (default: 1)

Returns a Vector of length R with f[n] = oscillator strength for root n.
f[ref_root] = 0 by convention.
"""
function compute_oscillator_strengths(E::AbstractVector{T},
                                       γ_aa::Array{T,4}, γ_bb::Array{T,4},
                                       dip_x::AbstractMatrix{T},
                                       dip_y::AbstractMatrix{T},
                                       dip_z::AbstractMatrix{T};
                                       ref_root::Int=1) where T
    R = length(E)
    size(γ_aa, 3) == R || throw(DimensionMismatch("E length $R ≠ γ root dimension"))

    tdm_x, tdm_y, tdm_z = compute_transition_dipoles(γ_aa, γ_bb, dip_x, dip_y, dip_z;
                                                      ref_root=ref_root)
    f = zeros(T, R)
    for n in 1:R
        n == ref_root && continue
        ΔE = E[n] - E[ref_root]
        ΔE > 0 || continue   # skip if reference is not the lowest
        f[n] = (2/3) * ΔE * (tdm_x[n]^2 + tdm_y[n]^2 + tdm_z[n]^2)
    end
    return f
end


"""
    absorption_spectrum(E, f; ref_root=1, σ=0.005, npoints=2000,
                        ω_min=nothing, ω_max=nothing, lineshape=:lorentzian,
                        thresh=1e-10)

Compute an absorption spectrum by broadening the stick spectrum of
oscillator strengths `f` at excitation energies `E[n]-E[ref_root]`.

Arguments
---------
- `E`        : Vector of R eigenenergies (Hartree)
- `f`        : Vector of R oscillator strengths from `compute_oscillator_strengths`
- `ref_root` : index of ground state (default 1)
- `σ`        : broadening parameter (Hartree); half-width for Lorentzian,
               standard deviation for Gaussian (default 0.005 ≈ 0.14 eV)
- `npoints`  : number of frequency grid points (default 2000)
- `ω_min/max`: frequency range; defaults to transitions ± 5σ
- `lineshape`: `:lorentzian` (default) or `:gaussian`
- `thresh`   : minimum oscillator strength to include (default 1e-10);
               filters out numerical noise from symmetry-forbidden transitions

Returns `(ω, I)` where ω is the frequency grid (Hartree) and I is the
intensity (arbitrary units, proportional to molar extinction coefficient).
Returns `(T[], T[])` if no transitions exceed `thresh`.
"""
function absorption_spectrum(E::AbstractVector{T}, f::AbstractVector{T};
                              ref_root::Int=1,
                              σ::Real=0.005,
                              npoints::Int=2000,
                              ω_min=nothing,
                              ω_max=nothing,
                              lineshape::Symbol=:lorentzian,
                              thresh::Real=1e-10) where T

    R      = length(E)
    E0     = E[ref_root]
    ω_trans = [E[n] - E0 for n in 1:R if n != ref_root && f[n] > thresh]

    isempty(ω_trans) && return (T[], T[])

    _ωmin = isnothing(ω_min) ? max(0.0, minimum(ω_trans) - 5*σ) : Float64(ω_min)
    _ωmax = isnothing(ω_max) ? maximum(ω_trans) + 5*σ           : Float64(ω_max)

    ω = collect(range(T(_ωmin), T(_ωmax), length=npoints))
    I = zeros(T, npoints)

    for n in 1:R
        n == ref_root && continue
        f[n] > thresh || continue
        ω_n = E[n] - E0
        for (i, ω_i) in enumerate(ω)
            if lineshape == :lorentzian
                I[i] += f[n] * (σ / T(π)) / ((ω_i - ω_n)^2 + σ^2)
            elseif lineshape == :gaussian
                I[i] += f[n] * exp(-(ω_i - ω_n)^2 / (2*σ^2)) / (σ * sqrt(2*T(π)))
            else
                error("Unknown lineshape: $lineshape. Use :lorentzian or :gaussian")
            end
        end
    end

    return ω, I
end


"""
    print_stick_spectrum(E, f; ref_root=1, units=:hartree)

Print a human-readable stick spectrum table.

Set `units=:ev` to convert excitation energies to eV (1 Hartree = 27.2114 eV).
"""
function print_stick_spectrum(E::AbstractVector, f::AbstractVector;
                               ref_root::Int=1, units::Symbol=:hartree)
    conv = units == :ev ? 27.2114 : 1.0
    unit_str = units == :ev ? "eV" : "Ha"
    E0 = E[ref_root]

    @printf("\n  %-6s  %-14s  %-14s\n", "Root", "ΔE ($unit_str)", "f (osc. str.)")
    @printf("  %s\n", "-"^42)
    for n in eachindex(E)
        n == ref_root && continue
        @printf("  %-6i  %-14.6f  %-14.8f\n", n, (E[n]-E0)*conv, f[n])
    end
    println()
end
