using Moshi.Match: @match
using Moshi.Data: @data
using Moshi.Derive: @derive

@data BinaryTree{T} begin
    Leaf(T)
    struct Node
        data :: T
        left :: BinaryTree.Type{T}
        right :: BinaryTree.Type{T}
    end
end
@derive BinaryTree[Hash, Eq, Show]

function childrens(node::BinaryTree.Type) 
    @match node begin
        BinaryTree.Leaf => ()
        BinaryTree.Node(_, left, right) => (left, right)
    end
end

struct PartitionTree
    tree :: BinaryTree.Type{UnitRange{Int}}
end

@testitem "Test BinaryTree" begin
    using NonEquilibriumGreenFunction.BinaryTree
    left = BinaryTree.Leaf(1)
    right = BinaryTree.Leaf(2)
    node = BinaryTree.Node(42, left, right)
    @test node.data == 42
    @test NonEquilibriumGreenFunction.childrens(node) == (left, right)
end

function PartitionTree(range::UnitRange{Int})
    if length(range) <= 1
        return PartitionTree(BinaryTree.Leaf{UnitRange{Int}}(range))
    end
    idx_mid = length(range) ÷ 2
    mid = range[idx_mid]
    left = PartitionTree(range.start:mid)
    right = PartitionTree(mid+1:range.stop)
    return PartitionTree(BinaryTree.Node(range,left.tree,right.tree))
end

function PartitionTree(range::Number)
    return PartitionTree(range:range)
end

function split_partition(tree::PartitionTree)
    @match tree.tree begin
        BinaryTree.Leaf(_) => throw("Bottom of the partition three reached")
        BinaryTree.Node(data,left,right) => return (PartitionTree(left),PartitionTree(right))
    end
end

function get_range(tree::PartitionTree)
    @match tree.tree begin
        BinaryTree.Leaf(range) => range
        BinaryTree.Node(range, _, _) => range
    end
end

function isleaf(tree::PartitionTree)
    @match tree.tree begin
        BinaryTree.Leaf(_) => true
        _ => false
    end
end

@testitem "Test PartitionTree" begin
    import NonEquilibriumGreenFunction.PartitionTree
    a =  PartitionTree(1:3)
    left,right = NonEquilibriumGreenFunction.split_partition(a)
    b = PartitionTree(NonEquilibriumGreenFunction.BinaryTree.Leaf(1:1))
    c = PartitionTree(NonEquilibriumGreenFunction.BinaryTree.Leaf(3:3))
    @test left == PartitionTree(NonEquilibriumGreenFunction.BinaryTree.Leaf(1:1))
    @test right == PartitionTree(2:3)
    @test NonEquilibriumGreenFunction.isleaf(PartitionTree(1))
    @test !NonEquilibriumGreenFunction.isleaf(PartitionTree(1:2))
end
