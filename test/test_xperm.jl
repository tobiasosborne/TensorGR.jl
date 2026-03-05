using TensorGR: Perm, perm_identity, perm_compose, perm_inverse, perm_is_identity,
                xperm_schreier_sims, xperm_canonical_perm

@testset "Perm basics" begin
    id = perm_identity(4)
    @test id.data == Int32[1, 2, 3, 4]
    @test perm_is_identity(id)

    t12 = Perm(Int32[2, 1, 3, 4])
    @test !perm_is_identity(t12)
    @test t12.data[1] == 2
    @test t12.data[2] == 1
end

@testset "Perm composition" begin
    # perm_compose(p1, p2) = p2 ∘ p1 (apply p1 first, then p2)
    t12 = Perm(Int32[2, 1, 3, 4])   # (1 2)
    t23 = Perm(Int32[1, 3, 2, 4])   # (2 3)
    # (1 2) then (2 3): 1→2→3, 2→1→1, 3→3→2 = [3, 1, 2, 4]
    p = perm_compose(t12, t23)
    @test p.data == Int32[3, 1, 2, 4]

    id = perm_identity(4)
    @test perm_compose(t12, id).data == t12.data
    @test perm_compose(id, t12).data == t12.data
end

@testset "Perm inverse" begin
    t12 = Perm(Int32[2, 1, 3, 4])
    @test perm_compose(t12, perm_inverse(t12)) |> perm_is_identity

    cycle = Perm(Int32[2, 3, 1, 4])
    inv = perm_inverse(cycle)
    @test inv.data == Int32[3, 1, 2, 4]
    @test perm_compose(cycle, inv) |> perm_is_identity
end

@testset "Schreier-Sims on S3" begin
    n = 3
    g1 = Perm(Int32[2, 1, 3])
    g2 = Perm(Int32[2, 3, 1])
    base_hint = Int32[1, 2]

    newbase, sgs = xperm_schreier_sims(base_hint, [g1, g2], n)
    @test length(newbase) >= 2
    @test length(sgs) >= 2
end

@testset "Schreier-Sims on Z2" begin
    n = 2
    g1 = Perm(Int32[2, 1])
    base_hint = Int32[1]

    newbase, sgs = xperm_schreier_sims(base_hint, [g1], n)
    @test length(newbase) >= 1
    @test length(sgs) >= 1
end

@testset "canonical_perm simple antisymmetry" begin
    # Rank-2 antisymmetric tensor T_{ab}
    # n = 2 slots + 2 sign bits = 4
    # Generator: swap slots 1,2 and flip sign (swap 3,4)
    # Free index names: [1, 2]
    n = 4
    gen = Perm(Int32[2, 1, 4, 3])
    base = Int32[1, 2]
    freeps = Int32[1, 2]

    # T_{ab} already canonical
    perm_ab = Perm(Int32[1, 2, 3, 4])
    result = xperm_canonical_perm(perm_ab, base, [gen], freeps, Int32[], n)
    @test result.data == Int32[1, 2, 3, 4]

    # T_{ba} → -T_{ab} (sign flipped)
    perm_ba = Perm(Int32[2, 1, 3, 4])
    result2 = xperm_canonical_perm(perm_ba, base, [gen], freeps, Int32[], n)
    @test result2.data == Int32[1, 2, 4, 3]
end

@testset "canonical_perm Riemann symmetry" begin
    # R_{abcd}: 4 slots + 2 sign = 6 points
    # Generators: antisym(a,b), antisym(c,d), pair sym
    # All 4 indices are free
    n = 6
    g1 = Perm(Int32[2, 1, 3, 4, 6, 5])  # antisym (a,b)
    g2 = Perm(Int32[1, 2, 4, 3, 6, 5])  # antisym (c,d)
    g3 = Perm(Int32[3, 4, 1, 2, 5, 6])  # pair sym
    base = Int32[1, 2, 3, 4]
    gens = [g1, g2, g3]
    freeps = Int32[1, 2, 3, 4]

    # R_{abcd} = identity: canonical
    r1 = xperm_canonical_perm(Perm(Int32[1,2,3,4,5,6]), base, gens, freeps, Int32[], n)
    @test r1.data == Int32[1, 2, 3, 4, 5, 6]

    # R_{bacd} → -R_{abcd}
    r2 = xperm_canonical_perm(Perm(Int32[2,1,3,4,5,6]), base, gens, freeps, Int32[], n)
    @test r2.data == Int32[1, 2, 3, 4, 6, 5]

    # R_{abdc} → -R_{abcd}
    r3 = xperm_canonical_perm(Perm(Int32[1,2,4,3,5,6]), base, gens, freeps, Int32[], n)
    @test r3.data == Int32[1, 2, 3, 4, 6, 5]

    # R_{cdab} → R_{abcd} (pair symmetry, no sign change)
    r4 = xperm_canonical_perm(Perm(Int32[3,4,1,2,5,6]), base, gens, freeps, Int32[], n)
    @test r4.data == Int32[1, 2, 3, 4, 5, 6]

    # R_{badc} → two antisymmetries = no sign change
    r5 = xperm_canonical_perm(Perm(Int32[2,1,4,3,5,6]), base, gens, freeps, Int32[], n)
    @test r5.data == Int32[1, 2, 3, 4, 5, 6]

    # R_{dcba} → pair sym + two antisym = no sign change
    r6 = xperm_canonical_perm(Perm(Int32[4,3,2,1,5,6]), base, gens, freeps, Int32[], n)
    @test r6.data == Int32[1, 2, 3, 4, 5, 6]

    # R_{dcab} → pair sym + antisym(c,d) = -R_{abcd}
    r7 = xperm_canonical_perm(Perm(Int32[3,4,2,1,5,6]), base, gens, freeps, Int32[], n)
    @test r7.data == Int32[1, 2, 3, 4, 6, 5]
end

@testset "canonical_perm with dummy indices" begin
    # g^{ab} T_{cd} with T antisymmetric, contracted a↔c and b↔d
    # Slots: 1=a(up from g), 2=b(up from g), 3=c(down from T), 4=d(down from T), 5=+, 6=-
    # Symmetry generators:
    #   g symmetric: [2,1,3,4,5,6]
    #   T antisymmetric: [1,2,4,3,6,5]
    # Dummy pairs: names (1,3) and (2,4) → dummyps = [1,3,2,4]
    # No free indices
    n = 6
    g_sym = Perm(Int32[2, 1, 3, 4, 5, 6])
    T_antisym = Perm(Int32[1, 2, 4, 3, 6, 5])
    base = Int32[1, 2, 3, 4]
    gens = [g_sym, T_antisym]
    freeps = Int32[]
    dummyps = Int32[1, 3, 2, 4]

    # The identity perm [1,2,3,4,5,6] represents g^{ab} T_{ab}
    perm = Perm(Int32[1, 2, 3, 4, 5, 6])
    r = xperm_canonical_perm(perm, base, gens, freeps, dummyps, n)
    # g^{ab} T_{ab} = 0 for T antisymmetric, g symmetric
    # xperm.c signals zero by returning all zeros
    @test r.data == Int32[0, 0, 0, 0, 0, 0]
end
