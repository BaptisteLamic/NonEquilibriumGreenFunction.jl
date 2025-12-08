using Moshi.Match: @match
using Moshi.Data: @data
using Moshi.Derive: @derive
using StacticArrays

@data StaticHodlr{N} begin
    struct Leaf{T}
end
@derive SymbolicExpression[Hash, Eq, Show]