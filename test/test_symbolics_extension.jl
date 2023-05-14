


@testitem "Symbolics kernel simplify null product" begin
    using Symbolics
    @variables x::Kernel y::Kernel z::Kernel
    @test simplify_kernel(0 * x) == 0
    @test simplify_kernel(x * 0) == 0
end


@testitem "Symbolics kernel simplify sum by zero" begin
    using Symbolics
    @variables x::Kernel y::Kernel z::Kernel
    @test simplify_kernel( 0 + y ) isa SymbolicUtils.BasicSymbolic{Kernel}
    @test simplify_kernel( 0 + y) == y
    @test simplify_kernel(x - 0) == x
    @test simplify_kernel(0 - y) isa SymbolicUtils.BasicSymbolic{Kernel}
end
