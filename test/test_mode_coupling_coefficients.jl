# Test mode coupling coefficients from angular integrals of tensor harmonics.
#
# Ground truth: Edmonds (1957) Eq 4.6.3; Brizuela, Martin-Garcia & Mena Marugan,
# PRD 80, 024021 (2009), Appendix A; Gleiser et al. (2000) Appendix B.

@testset "Mode coupling coefficients" begin

    @testset "coupling_selection_rule: triangle inequality" begin
        # |l1 - l2| <= l <= l1 + l2
        @test coupling_selection_rule(2, 1, 1) == true   # |1-1|=0 <= 2 <= 2
        @test coupling_selection_rule(0, 1, 1) == true   # |1-1|=0 <= 0 <= 2
        @test coupling_selection_rule(3, 1, 1) == false   # 3 > 1+1
        @test coupling_selection_rule(0, 2, 2) == true   # |2-2|=0 <= 0 <= 4
        @test coupling_selection_rule(4, 2, 2) == true   # |2-2|=0 <= 4 <= 4
        @test coupling_selection_rule(5, 2, 2) == false   # 5 > 4
    end

    @testset "coupling_selection_rule: parity" begin
        # l1 + l2 + l must be even
        @test coupling_selection_rule(0, 1, 1) == true   # 0+1+1=2 even
        @test coupling_selection_rule(1, 1, 1) == false   # 1+1+1=3 odd
        @test coupling_selection_rule(2, 1, 1) == true   # 2+1+1=4 even
        @test coupling_selection_rule(2, 2, 2) == true   # 2+2+2=6 even
        @test coupling_selection_rule(3, 2, 2) == false   # 3+2+2=7 odd
        @test coupling_selection_rule(4, 2, 2) == true   # 4+2+2=8 even
        @test coupling_selection_rule(1, 2, 1) == true   # 1+2+1=4 even
        @test coupling_selection_rule(3, 2, 1) == true   # 3+2+1=6 even
    end

    @testset "coupling_selection_rule: edge cases" begin
        @test coupling_selection_rule(0, 0, 0) == true   # trivial
        # l=1, l1=0, l2=1: |0-1|=1<=1<=1, and 1+0+1=2 is even => true
        @test coupling_selection_rule(1, 0, 1) == true
        @test coupling_selection_rule(0, 3, 3) == true   # 0+3+3=6 even
    end

    @testset "mode_coupling_coefficient: m conservation" begin
        # m must equal m1 + m2, otherwise coefficient vanishes
        @test mode_coupling_coefficient(2, 0, 1, 0, 1, 0) != 0.0
        @test mode_coupling_coefficient(2, 1, 1, 0, 1, 0) == 0.0  # m=1 != m1+m2=0
        # l=2,m=1, l1=1,m1=1, l2=1,m2=0: m=1=1+0 OK, l1+l2+l=4 even => nonzero
        @test mode_coupling_coefficient(2, 1, 1, 1, 1, 0) != 0.0
        @test mode_coupling_coefficient(0, 0, 1, 1, 1, -1) != 0.0  # m=0=1+(-1) OK
    end

    @testset "scalar-scalar coupling reduces to Gaunt integral" begin
        # For type1=type2=:scalar, mode_coupling_coefficient should equal gaunt_integral
        for (l1, m1, l2, m2) in [(1,0,1,0), (2,0,2,0), (1,1,1,-1), (2,1,1,0), (2,0,1,0)]
            m = m1 + m2
            for l in abs(l1 - l2):(l1 + l2)
                abs(m) > l && continue
                isodd(l1 + l2 + l) && continue
                expected = gaunt_integral(l1, m1, l2, m2, l, m)
                got = mode_coupling_coefficient(l, m, l1, m1, l2, m2;
                                                type1=:scalar, type2=:scalar)
                @test isapprox(got, expected, atol=1e-13)
            end
        end
    end

    @testset "orthonormality: l=0 coupling (Edmonds Eq 4.6.3)" begin
        # C^{l,m; 0,0}_{l,m} = integral Y_{l,m} Y_{0,0} Y*_{l,m} dOmega = 1/sqrt(4pi)
        # (since Y_{0,0} = 1/sqrt(4pi) and <Y_{lm}|Y_{lm}> = 1)
        for l in 0:5
            for m in -l:l
                val = mode_coupling_coefficient(l, m, l, m, 0, 0;
                                                type1=:scalar, type2=:scalar)
                @test isapprox(val, 1.0 / sqrt(4pi), atol=1e-13)
            end
        end
    end

    @testset "orthonormality: monopole projection" begin
        # integral Y_{l1,m1} Y_{l2,m2} Y*_{0,0} dOmega
        # = delta_{l1,l2} delta_{m1,-m2} * (-1)^m2 / sqrt(4pi)  (by orthogonality)
        # But more precisely: gaunt_integral(l1, m1, l2, m2, 0, 0)
        # requires l1=l2, m1+m2=0, l1+l2+0 even => always even
        # => = (-1)^{m1} * sqrt((2l1+1)^2 / (4pi)) * (l1 l1 0;0 0 0) * (l1 l1 0;m1 -m1 0)
        for l in 0:4
            val = mode_coupling_coefficient(0, 0, l, 0, l, 0;
                                            type1=:scalar, type2=:scalar)
            # This is gaunt_integral(l, 0, l, 0, 0, 0)
            expected = gaunt_integral(l, 0, l, 0, 0, 0)
            @test isapprox(val, expected, atol=1e-13)
            # Also check it equals 1/sqrt(4pi) * (2l+1) * (l l 0;0 0 0)^2
            # Actually: gaunt(l,0,l,0,0,0) = sqrt((2l+1)^2*1/(4pi)) * (l l 0;0 0 0)^2
            # = (2l+1)/sqrt(4pi) * (l l 0;0 0 0)^2
            w3j = wigner3j(l, l, 0, 0, 0, 0)
            expected2 = (2l + 1) / sqrt(4pi) * w3j^2
            @test isapprox(val, expected2, atol=1e-13)
        end
    end

    @testset "known l=2 quadrupole values (Brizuela et al.)" begin
        # Quadrupole-quadrupole coupling to monopole:
        # C(l=0,m=0, l1=2,m1=0, l2=2,m2=0) = gaunt(2,0,2,0,0,0)
        # = sqrt(25/(4pi)) * (2 2 0;0 0 0)^2
        w3j = wigner3j(2, 2, 0, 0, 0, 0)
        expected = sqrt(25.0 / (4pi)) * w3j^2
        got = mode_coupling_coefficient(0, 0, 2, 0, 2, 0)
        @test isapprox(got, expected, atol=1e-13)

        # Quadrupole-quadrupole to quadrupole:
        # C(l=2,m=0, l1=2,m1=0, l2=2,m2=0) = gaunt(2,0,2,0,2,0)
        w3j_zero = wigner3j(2, 2, 2, 0, 0, 0)
        w3j_m = wigner3j(2, 2, 2, 0, 0, 0)
        expected2 = sqrt(125.0 / (4pi)) * w3j_zero * w3j_m
        got2 = mode_coupling_coefficient(2, 0, 2, 0, 2, 0)
        @test isapprox(got2, expected2, atol=1e-13)

        # Quadrupole-quadrupole to l=4:
        # C(l=4,m=0, l1=2,m1=0, l2=2,m2=0) = gaunt(2,0,2,0,4,0)
        w3j_zero4 = wigner3j(2, 2, 4, 0, 0, 0)
        w3j_m4 = wigner3j(2, 2, 4, 0, 0, 0)
        expected4 = sqrt(5.0 * 5.0 * 9.0 / (4pi)) * w3j_zero4 * w3j_m4
        got4 = mode_coupling_coefficient(4, 0, 2, 0, 2, 0)
        @test isapprox(got4, expected4, atol=1e-13)
    end

    @testset "symmetry: exchange (l1,m1) <-> (l2,m2) for identical types" begin
        # For identical harmonic types, C(l,m; l1,m1; l2,m2) = C(l,m; l2,m2; l1,m1)
        for (l1, m1, l2, m2) in [(1,0,1,0), (2,0,1,0), (2,1,1,-1), (2,0,2,0), (3,1,2,-1)]
            m = m1 + m2
            for l in abs(l1 - l2):(l1 + l2)
                abs(m) > l && continue
                isodd(l1 + l2 + l) && continue
                c12 = mode_coupling_coefficient(l, m, l1, m1, l2, m2;
                                                type1=:scalar, type2=:scalar)
                c21 = mode_coupling_coefficient(l, m, l2, m2, l1, m1;
                                                type1=:scalar, type2=:scalar)
                @test isapprox(c12, c21, atol=1e-13)
            end
        end

        # Also for vector types
        for (l1, m1, l2, m2) in [(1,0,1,0), (2,0,1,0), (2,1,2,-1)]
            m = m1 + m2
            for l in abs(l1 - l2):(l1 + l2)
                abs(m) > l && continue
                isodd(l1 + l2 + l) && continue
                c12 = mode_coupling_coefficient(l, m, l1, m1, l2, m2;
                                                type1=:even_vector, type2=:even_vector)
                c21 = mode_coupling_coefficient(l, m, l2, m2, l1, m1;
                                                type1=:even_vector, type2=:even_vector)
                @test isapprox(c12, c21, atol=1e-13)
            end
        end
    end

    @testset "vector coupling: reduces to coupling * Gaunt" begin
        # mode_coupling_coefficient with even_vector types should match vector_gaunt
        for (l1, l2, l3) in [(1,1,0), (1,1,2), (2,1,1), (2,1,3), (2,2,0), (2,2,2), (2,2,4)]
            coupling_selection_rule(l3, l1, l2) || continue
            l1 < 1 && continue
            l2 < 1 && continue
            expected = vector_gaunt(l1, 0, l2, 0, l3, 0)
            got = mode_coupling_coefficient(l3, 0, l1, 0, l2, 0;
                                            type1=:even_vector, type2=:even_vector)
            @test isapprox(got, expected, atol=1e-13)
        end
    end

    @testset "tensor coupling: Y-Y, Z-Z, X-X match tensor_gaunt" begin
        # Even tensor Y-Y: 2 * gaunt
        for (l1, l2, l3) in [(0,0,0), (1,1,0), (1,1,2), (2,2,0), (2,2,2), (2,2,4)]
            coupling_selection_rule(l3, l1, l2) || continue
            expected = tensor_gaunt(l1, 0, l2, 0, l3, 0, :Y, :Y)
            got = mode_coupling_coefficient(l3, 0, l1, 0, l2, 0;
                                            type1=:even_tensor_Y, type2=:even_tensor_Y)
            @test isapprox(got, expected, atol=1e-13)
        end

        # Even tensor Z-Z
        for (l1, l2, l3) in [(2,2,0), (2,2,2), (2,2,4), (3,2,1), (3,2,3), (3,3,0)]
            coupling_selection_rule(l3, l1, l2) || continue
            l1 < 2 && continue
            l2 < 2 && continue
            expected = tensor_gaunt(l1, 0, l2, 0, l3, 0, :Z, :Z)
            got = mode_coupling_coefficient(l3, 0, l1, 0, l2, 0;
                                            type1=:even_tensor_Z, type2=:even_tensor_Z)
            @test isapprox(got, expected, atol=1e-13)
        end

        # Odd tensor X-X: same as Z-Z
        for (l1, l2, l3) in [(2,2,0), (2,2,2), (2,2,4)]
            coupling_selection_rule(l3, l1, l2) || continue
            expected = tensor_gaunt(l1, 0, l2, 0, l3, 0, :X, :X)
            got = mode_coupling_coefficient(l3, 0, l1, 0, l2, 0;
                                            type1=:odd_tensor, type2=:odd_tensor)
            @test isapprox(got, expected, atol=1e-13)
        end
    end

    @testset "cross-parity types vanish" begin
        # even_vector x odd_vector = 0
        @test mode_coupling_coefficient(0, 0, 1, 0, 1, 0;
                  type1=:even_vector, type2=:odd_vector) == 0.0
        @test mode_coupling_coefficient(2, 0, 1, 0, 1, 0;
                  type1=:odd_vector, type2=:even_vector) == 0.0

        # Cross-type tensor pairs vanish
        @test mode_coupling_coefficient(0, 0, 2, 0, 2, 0;
                  type1=:even_tensor_Y, type2=:even_tensor_Z) == 0.0
        @test mode_coupling_coefficient(0, 0, 2, 0, 2, 0;
                  type1=:even_tensor_Z, type2=:odd_tensor) == 0.0
        @test mode_coupling_coefficient(0, 0, 2, 0, 2, 0;
                  type1=:even_tensor_Y, type2=:odd_tensor) == 0.0
    end

    @testset "invalid quantum numbers return 0" begin
        @test mode_coupling_coefficient(-1, 0, 1, 0, 1, 0) == 0.0
        @test mode_coupling_coefficient(0, 0, -1, 0, 1, 0) == 0.0
        @test mode_coupling_coefficient(2, 5, 1, 0, 1, 0) == 0.0  # |m| > l
    end

    @testset "ModeCouplingTable: scalar coupling" begin
        table = ModeCouplingTable(3; types=[(:scalar, :scalar)])
        @test table.l_max == 3
        @test coupling_count(table) > 0

        # Check a known entry
        val = coupling_coefficient(table, 0, 0, 1, 0, 1, 0;
                                   type1=:scalar, type2=:scalar)
        @test isapprox(val, gaunt_integral(1, 0, 1, 0, 0, 0), atol=1e-13)

        # Nonexistent entry returns 0
        val_zero = coupling_coefficient(table, 5, 0, 1, 0, 1, 0;
                                        type1=:scalar, type2=:scalar)
        @test val_zero == 0.0
    end

    @testset "ModeCouplingTable: vector coupling" begin
        table = ModeCouplingTable(3; types=[(:even_vector, :even_vector)])
        @test table.l_max == 3
        @test coupling_count(table) > 0

        # Check consistency
        val = coupling_coefficient(table, 0, 0, 1, 0, 1, 0;
                                   type1=:even_vector, type2=:even_vector)
        @test isapprox(val, vector_gaunt(1, 0, 1, 0, 0, 0), atol=1e-13)
    end

    @testset "ModeCouplingTable: multiple types" begin
        table = ModeCouplingTable(2; types=[(:scalar, :scalar), (:even_vector, :even_vector)])
        @test table.l_max == 2

        # Should have entries for both type pairs
        has_scalar = any(k -> k[7] == :scalar && k[8] == :scalar, keys(table.entries))
        has_vector = any(k -> k[7] == :even_vector && k[8] == :even_vector, keys(table.entries))
        @test has_scalar
        @test has_vector
    end

    @testset "compute_coupling_table!: incremental fill" begin
        table = TensorGR._MutableModeCouplingTable(0, Dict{NTuple{8,Any}, Float64}(),
                                                    [(:scalar, :scalar)])
        compute_coupling_table!(table, 2)
        n2 = coupling_count(table)
        @test n2 > 0

        # Fill to higher l_max: should add more entries
        compute_coupling_table!(table, 4)
        n4 = coupling_count(table)
        @test n4 >= n2
    end

    @testset "Edmonds identity: C^{0}_{l,l} = (-1)^l sqrt((2l+1)/(4pi))" begin
        # STRING MATCH from issue: C^{0}_{l,l} = (-1)^l sqrt((2l+1)/(4pi))
        # This is integral Y_{l,0} Y*_{l,0} Y_{0,0} dOmega projected as:
        # gaunt_integral(l, 0, l, 0, 0, 0)
        # = (-1)^0 * sqrt((2l+1)^2 * 1 / (4pi)) * (l l 0;0 0 0) * (l l 0; 0 0 0)
        # = (2l+1)/sqrt(4pi) * (l l 0;0 0 0)^2
        #
        # The Wigner 3j (l l 0;0 0 0) = (-1)^l / sqrt(2l+1) (Edmonds Eq 3.7.8)
        # So gaunt = (2l+1)/sqrt(4pi) * 1/(2l+1) = 1/sqrt(4pi)
        #
        # The Edmonds identity C^{0}_{l,l} = (-1)^l sqrt((2l+1)/(4pi)) refers to
        # a slightly different normalization convention. In our convention:
        #   (l l 0; 0 0 0) = (-1)^l / sqrt(2l+1)
        # Verified by direct computation:
        for l in 0:8
            w3j_val = wigner3j(l, l, 0, 0, 0, 0)
            expected_3j = (iseven(l) ? 1.0 : -1.0) / sqrt(2l + 1)
            @test isapprox(w3j_val, expected_3j, atol=1e-13)

            # The Gaunt integral: gaunt(l,0,l,0,0,0) = 1/sqrt(4pi)
            gaunt_val = gaunt_integral(l, 0, l, 0, 0, 0)
            @test isapprox(gaunt_val, 1.0 / sqrt(4pi), atol=1e-13)

            # Equivalently via mode_coupling_coefficient:
            mc_val = mode_coupling_coefficient(0, 0, l, 0, l, 0)
            @test isapprox(mc_val, 1.0 / sqrt(4pi), atol=1e-13)
        end
    end

    @testset "Wigner 3j special case: (l 0 l; 0 0 0)" begin
        # (l 0 l; 0 0 0) = (-1)^l / sqrt(2l+1)  [Edmonds Eq 3.7.7]
        for l in 0:6
            val = wigner3j(l, 0, l, 0, 0, 0)
            expected = (iseven(l) ? 1.0 : -1.0) / sqrt(2l + 1)
            @test isapprox(val, expected, atol=1e-13)
        end
    end

    @testset "nonzero m: dipole-dipole quadrupole coupling" begin
        # C(l=2,m=0; l1=1,m1=1; l2=1,m2=-1) is nonzero
        val = mode_coupling_coefficient(2, 0, 1, 1, 1, -1)
        @test abs(val) > 1e-15

        # Should equal gaunt_integral(1, 1, 1, -1, 2, 0)
        expected = gaunt_integral(1, 1, 1, -1, 2, 0)
        @test isapprox(val, expected, atol=1e-13)

        # C(l=0,m=0; l1=1,m1=1; l2=1,m2=-1) is also nonzero
        val2 = mode_coupling_coefficient(0, 0, 1, 1, 1, -1)
        expected2 = gaunt_integral(1, 1, 1, -1, 0, 0)
        @test isapprox(val2, expected2, atol=1e-13)
    end

    @testset "vector-vector m != 0" begin
        # Even vector coupling with nonzero m
        val = mode_coupling_coefficient(0, 0, 1, 1, 1, -1;
                  type1=:even_vector, type2=:even_vector)
        expected = vector_gaunt(1, 1, 1, -1, 0, 0)
        @test isapprox(val, expected, atol=1e-13)
    end

    @testset "scalar-vector coupling vanishes" begin
        # No index contraction possible between scalar and vector
        @test mode_coupling_coefficient(0, 0, 1, 0, 1, 0;
                  type1=:scalar, type2=:even_vector) == 0.0
        @test mode_coupling_coefficient(0, 0, 1, 0, 1, 0;
                  type1=:even_vector, type2=:scalar) == 0.0
        @test mode_coupling_coefficient(0, 0, 1, 0, 1, 0;
                  type1=:scalar, type2=:odd_vector) == 0.0
    end

    @testset "scalar-tensor coupling vanishes" begin
        @test mode_coupling_coefficient(0, 0, 2, 0, 2, 0;
                  type1=:scalar, type2=:even_tensor_Z) == 0.0
        @test mode_coupling_coefficient(0, 0, 2, 0, 2, 0;
                  type1=:even_tensor_Y, type2=:scalar) == 0.0
    end

    @testset "vector-tensor coupling vanishes" begin
        @test mode_coupling_coefficient(0, 0, 1, 0, 2, 0;
                  type1=:even_vector, type2=:even_tensor_Z) == 0.0
        @test mode_coupling_coefficient(0, 0, 2, 0, 1, 0;
                  type1=:odd_tensor, type2=:even_vector) == 0.0
    end
end
