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

@testitem "Test PartitionTree properly make a partition of the domain" begin
    import NonEquilibriumGreenFunction: build_partition, PartitionTree, collect_leaf_data
    for leaf_size in 1:12
        for n in 1:12
            partition_tree = build_partition(1:n, leaf_size)
            all_indices = collect_leaf_data(partition_tree.tree)
            @test all_indices == collect(1:n)
        end
    end
end