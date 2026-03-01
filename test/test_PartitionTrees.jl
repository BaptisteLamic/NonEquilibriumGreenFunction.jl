@testitem "Test BinaryTree" begin
    using NonEquilibriumGreenFunction.BinaryTree
    left = BinaryTree.Leaf(1)
    right = BinaryTree.Leaf(2)
    node = BinaryTree.Node(42, left, right)
    @test node.data == 42
    @test NonEquilibriumGreenFunction.childrens(node) == (left, right)
end

@testitem "Test PartitionTree" begin
    import NonEquilibriumGreenFunction: build_partition, PartitionTree
    a = build_partition(1:3, 1)
    left, right = NonEquilibriumGreenFunction.split_partition(a)
    b = PartitionTree(NonEquilibriumGreenFunction.BinaryTree.Leaf(1:1))
    c = PartitionTree(NonEquilibriumGreenFunction.BinaryTree.Leaf(3:3))
    @test left == PartitionTree(NonEquilibriumGreenFunction.BinaryTree.Leaf(1:1))
    @test right == build_partition(2:3, 1)
    @test NonEquilibriumGreenFunction.isleaf(build_partition(1, 1))
    @test !NonEquilibriumGreenFunction.isleaf(build_partition(1:2, 1))
end
