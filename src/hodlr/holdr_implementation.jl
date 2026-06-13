import Base: size, eltype, *, view, inv, get_index
export build_hodlr
export full

# T is the scalar type
struct LeafHodlr{T} <: HodlrTree{T}
    data::Matrix{T}
    settings::HodlrSettings
end
function LeafHodlr(data,settings)
    return LeafHodlr{eltype(data)}(data,settings)
end
struct NodeHodlr{T} <: HodlrTree{T}
    A::HodlrTree{T}
    B::HodlrTree{T}
    upper_offdiag::LowRankBlock{T}
    lower_offdiag::LowRankBlock{T}
    settings::HodlrSettings
end
function NodeHodlr(A::HodlrTree{T},B::HodlrTree{T},upper::LowRankBlock{T},lower::LowRankBlock{T}) where T
    settings = combine_settings(A.settings, B.settings)
    #GPT CODE START
    expected_upper = (size(A,1), size(B,2))
    expected_lower = (size(B,1), size(A,2))
    upper2 = upper
    lower2 = lower
    if size(upper2) != expected_upper
        Ufull = full(upper2)
        if size(Ufull) == (expected_upper[2], expected_upper[1])
            Utrim = view(Ufull, 1:expected_upper[2], 1:expected_upper[1])
            upper2 = SvdBlock(Matrix(transpose(Matrix(Utrim))), settings)
        else
            Utrim = view(Ufull, 1:expected_upper[1], 1:expected_upper[2])
            upper2 = SvdBlock(Matrix(Utrim), settings)
        end
    end
    if size(lower2) != expected_lower
        Lfull = full(lower2)
        if size(Lfull) == (expected_lower[2], expected_lower[1])
            Ltrim = view(Lfull, 1:expected_lower[2], 1:expected_lower[1])
            lower2 = SvdBlock(Matrix(transpose(Matrix(Ltrim))), settings)
        else
            Ltrim = view(Lfull, 1:expected_lower[1], 1:expected_lower[2])
            lower2 = SvdBlock(Matrix(Ltrim), settings)
        end
    end
    #GPT CODE END
    return NodeHodlr{T}(A,B,upper2,lower2,settings)
end

function isleaf(matrix::HodlrTree)
    return isa(matrix, LeafHodlr)
end

function isnode(matrix::HodlrTree)
    return isa(matrix, NodeHodlr)
end

function get_children(matrix::HodlrTree)
    if isleaf(matrix)
        return matrix.data
    elseif isnode(matrix)
        return (matrix.A, matrix.B, matrix.upper_offdiag, matrix.lower_offdiag)
    else
        throw(ArgumentError("Matrix is neither a leaf nor a node"))
    end
end

function _convert_matrix_partition_to_domain(partition_row::PartitionTree, partition_col::PartitionTree, domain, bs)
    range = get_range(partition_row)
    converted_partition_row = div(range[1] - 1, bs)+1:div(range[end] - 1, bs)+1
    range = get_range(partition_col)
    converted_partition_col = div(range[1] - 1, bs)+1:div(range[end] - 1, bs)+1
    raw_domain_row = xaxis(domain)[converted_partition_row]
    raw_domain_col = yaxis(domain)[converted_partition_col]
    return KernelDomain((raw_domain_row[1], raw_domain_row[end]), (raw_domain_col[1], raw_domain_col[end]), x_steps=length(raw_domain_row), y_steps=length(raw_domain_col))
end

function build_hodlr(kf::KernelFunction, row_partition::PartitionTree, col_partition::PartitionTree, settings::HodlrSettings)
    if isleaf(row_partition) && isleaf(col_partition)
        return _construct_leaf(kf, row_partition, col_partition, settings)
    else
        upper_rows, lower_row = split_partition(row_partition)
        left_cols, right_cols = split_partition(col_partition)
        upper_offdiag_kernel = restrict_domain(kf,
            _convert_matrix_partition_to_domain(upper_rows, right_cols, kf.domain, blocksize(kf)))
        lower_offdiag_kernel = restrict_domain(kf,
            _convert_matrix_partition_to_domain(lower_row, left_cols, kf.domain, blocksize(kf)))

        A = build_hodlr(kf, upper_rows, left_cols, settings)
        B = build_hodlr(kf, lower_row, right_cols, settings)
        upper_offdiag = SvdBlock(upper_offdiag_kernel, settings)
        lower_offdiag = SvdBlock(lower_offdiag_kernel, settings)
        #GPT CODE START
        # Ensure off-diagonal blocks have the expected element sizes. If the
        # low-rank approximation returns a slightly larger block (due to block
        # rounding), trim it to the expected dimensions and rebuild a SvdBlock
        # from the trimmed dense block.
        expected_upper = (size(A,1), size(B,2))
        expected_lower = (size(B,1), size(A,2))
        if size(upper_offdiag) != expected_upper
            Ufull = full(upper_offdiag)
            Utrim = view(Ufull, 1:expected_upper[1], 1:expected_upper[2])
            upper_offdiag = SvdBlock(Matrix(Utrim), settings)
        end
        if size(lower_offdiag) != expected_lower
            Lfull = full(lower_offdiag)
            # If the low-rank block was produced with transposed shape, try
            # transposing the dense block before rebuilding. This recovers
            # cases where row/column axes were swapped earlier in the pipeline.
            if size(Lfull) == (expected_lower[2], expected_lower[1])
                Ltrim = view(Lfull, 1:expected_lower[2], 1:expected_lower[1])
                lower_offdiag = SvdBlock(Matrix(transpose(Matrix(Ltrim))), settings)
            else
                Ltrim = view(Lfull, 1:expected_lower[1], 1:expected_lower[2])
                lower_offdiag = SvdBlock(Matrix(Ltrim), settings)
            end
        end
        #GPT CODE END
        return NodeHodlr(A, B, upper_offdiag, lower_offdiag)
    end
end
function _construct_leaf(kf, row_partition::PartitionTree, col_partition::PartitionTree, settings)
    restricted_kf = restrict_domain(kf, _convert_matrix_partition_to_domain(row_partition, col_partition, kf.domain, blocksize(kf)))
    M = zeros(eltype(restricted_kf), size(restricted_kf, 1), size(restricted_kf, 2))
    fill_with_kernel!(M, restricted_kf)
    return LeafHodlr(M,settings)
end


function build_hodlr(kf::KernelFunction, ctx::HodlrSettings)
    row_partition = build_partition(1:size(kf, 1), ctx.leafsize)
    col_partition = build_partition(1:size(kf, 2), ctx.leafsize)
    return build_hodlr(kf, row_partition, col_partition, ctx)
end
function build_hodlr(matrix::Union{LowRankBlock,AbstractMatrix}, ctx::HodlrSettings)
    row_partition = build_partition(1:size(matrix, 1), ctx.leafsize)
    col_partition = build_partition(1:size(matrix, 2), ctx.leafsize)
    return _build_hodlr_from_lowrank(matrix, row_partition, col_partition, ctx)
end
function _build_hodlr_from_lowrank(matrix::Union{LowRankBlock,AbstractMatrix},row_partition, col_partition, settings::HodlrSettings)
    if isleaf(row_partition) && isleaf(col_partition)
        return _construct_leaf(matrix, row_partition, col_partition, settings)
    end
    upper_rows, lower_row = split_partition(row_partition)
    left_cols, right_cols = split_partition(col_partition)
    A = _build_hodlr_from_lowrank(matrix, upper_rows, left_cols, settings)
    B = _build_hodlr_from_lowrank(matrix, lower_row, right_cols, settings)
    upper_offdiag = _construct_offdiag_block(matrix, upper_rows, right_cols, settings)
    lower_offdiag = _construct_offdiag_block(matrix, lower_row, left_cols, settings)
    #GPT CODE START
    expected_upper = (size(A,1), size(B,2))
    expected_lower = (size(B,1), size(A,2))
    if size(upper_offdiag) != expected_upper
        Ufull = full(upper_offdiag)
        Utrim = view(Ufull, 1:expected_upper[1], 1:expected_upper[2])
        upper_offdiag = SvdBlock(Matrix(Utrim), settings)
    end
    if size(lower_offdiag) != expected_lower
        Lfull = full(lower_offdiag)
        if size(Lfull) == (expected_lower[2], expected_lower[1])
            Ltrim = view(Lfull, 1:expected_lower[2], 1:expected_lower[1])
            lower_offdiag = SvdBlock(Matrix(transpose(Matrix(Ltrim))), settings)
        else
            Ltrim = view(Lfull, 1:expected_lower[1], 1:expected_lower[2])
            lower_offdiag = SvdBlock(Matrix(Ltrim), settings)
        end
    end
    #GPT CODE END
    return NodeHodlr(A, B, upper_offdiag, lower_offdiag)
end
function build_upper_triangular_hodlr(matrix::Union{LowRankBlock,AbstractMatrix}, ctx::HodlrSettings)
    row_partition = build_partition(1:size(matrix, 1), ctx.leafsize)
    col_partition = build_partition(1:size(matrix, 2), ctx.leafsize)
    return _build_upper_triangular_hodlr_from_lowrank(matrix, row_partition, col_partition, ctx)
end
function _build_upper_triangular_hodlr_from_lowrank(matrix::Union{LowRankBlock,AbstractMatrix},row_partition, col_partition, settings::HodlrSettings)
    if isleaf(row_partition) && isleaf(col_partition)
        return _construct_leaf(matrix, row_partition, col_partition, settings)
    end
    upper_rows, lower_row = split_partition(row_partition)
    left_cols, right_cols = split_partition(col_partition)
    A = _build_upper_triangular_hodlr_from_lowrank(matrix, upper_rows, left_cols, settings)
    B = _build_upper_triangular_hodlr_from_lowrank(matrix, lower_row, right_cols, settings)
    upper_offdiag = _construct_offdiag_block(matrix, upper_rows, right_cols, settings)
    lower_offdiag = ZeroBlock{eltype(matrix)}(size(upper_offdiag))
    return NodeHodlr(A, B, upper_offdiag, lower_offdiag)
end
function _construct_offdiag_block(matrix::LowRankBlock, row_partition::PartitionTree, col_partition::PartitionTree, settings)
    return view(matrix, get_range(row_partition), get_range(col_partition))
end
function _construct_offdiag_block(matrix::AbstractMatrix, row_partition::PartitionTree, col_partition::PartitionTree, settings)
    sub_matrix = view(matrix, get_range(row_partition), get_range(col_partition))
    return SvdBlock(sub_matrix, settings)
end
function _construct_leaf(matrix::LowRankBlock, row_partition::PartitionTree, col_partition::PartitionTree, settings)
    sub_matrix = view(matrix, get_range(row_partition), get_range(col_partition))
    return LeafHodlr(full(sub_matrix), settings)
end
function _construct_leaf(matrix::AbstractMatrix, row_partition::PartitionTree, col_partition::PartitionTree, settings)
    sub_matrix = view(matrix, get_range(row_partition), get_range(col_partition))
    return LeafHodlr(sub_matrix, settings)
end

function size(Hodlr::LeafHodlr{M}) where M
    return size(Hodlr.data)
end

function size(Hodlr::NodeHodlr{M}) where M
    return (size(Hodlr.A, 1) + size(Hodlr.B, 1), size(Hodlr.A, 2) + size(Hodlr.B, 2))
end

function size(Hodlr::HodlrTree{M}, i) where M
    s = size(Hodlr)
    if i < 1 || i > 2
        return 1
    else
        return s[i]
    end
end

function eltype(::HodlrTree{M}) where M
    return eltype(M)
end

function getindex(Hodlr::LeafHodlr, i, j)
    return Hodlr.data[i, j]
end
function getindex(Hodlr::NodeHodlr, i, j)
    nA1, nA2 = size(Hodlr.A)
    if i <= nA1 && j <= nA2
        return getindex(Hodlr.A, i, j)
    elseif i > nA1 && j > nA2
        return getindex(Hodlr.B, i - nA1, j - nA2)
    elseif i <= nA1 && j > nA2
        return full(view(Hodlr.upper_offdiag, i, j - nA2))
    else
        return full(view(Hodlr.lower_offdiag, i - nA1, j))
    end
end

function full(Hodlr::LeafHodlr{M}) where M
    return Hodlr.data
end

function full(Hodlr::NodeHodlr{M}) where M
    dense_matrix = zeros(eltype(Hodlr), size(Hodlr)...)
    _full!(dense_matrix, Hodlr)
    return dense_matrix
end

function _full!(out, Hodlr::LeafHodlr)
    M = Hodlr.data
    out[:, :] .= M
end

function _full!(out, hodlr::NodeHodlr)
    A, B, upper_offdiag, lower_offdiag = hodlr.A, hodlr.B, hodlr.upper_offdiag, hodlr.lower_offdiag
    nA1, nA2 = size(A)
    out_up = view(out, 1:nA1, 1:nA2)
    n1,n2 = size(out)
    out_down = view(out, nA1+1:n1, nA2+1:n2)
    _full!(out_up, A)
    _full!(out_down, B)
    U = full(upper_offdiag)
    L = full(lower_offdiag)
    expected_up_cols = n2 - nA2
    expected_down_rows = n1 - nA1
    if size(U) != (nA1, expected_up_cols)
        throw(ErrorException(string("HODLR _full! mismatch: upper_offdiag full size ", size(U), " != target ", (nA1, expected_up_cols), "; A.size=", size(A), ", B.size=", size(B), ", upper_offdiag.size=", size(upper_offdiag), ", lower_offdiag.size=", size(lower_offdiag))))
    end
    if size(L) != (expected_down_rows, nA2)
        throw(ErrorException(string("HODLR _full! mismatch: lower_offdiag full size ", size(L), " != target ", (expected_down_rows, nA2), "; A.size=", size(A), ", B.size=", size(B), ", upper_offdiag.size=", size(upper_offdiag), ", lower_offdiag.size=", size(lower_offdiag))))
    end
    out[1:nA1, nA2+1:end] .= U
    out[nA1+1:end, 1:nA2] .= L
end

function (*)(left::T, x::M) where {T<:HodlrTree,M<:AbstractArray}
    _throw_error_if_incompatible_size(left, x)
    out = zeros(eltype(left), size(left, 1), size(x, 2))
    _apply_right_mul!(out, left, x)
    return out
end
function _apply_right_mul!(out, Hodlr::LeafHodlr, x)
    M = Hodlr.data
    out[:, :] .= M * x
end
function full(x::Array)
    return x
end
function _apply_right_mul!(out, Hodlr::NodeHodlr, x)
    A, B, upper_offdiag, lower_offdiag = Hodlr.A, Hodlr.B, Hodlr.upper_offdiag, Hodlr.lower_offdiag
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    x_up = view(x, 1:nA2, :)
    x_down = view(x, nA2+1:nA2+nB2, :)
    out_up = view(out, 1:nA1, :)
    out_down = view(out, nA1+1:nA1+nB1, :)
    _apply_right_mul!(out_up, A, x_up)
    _apply_right_mul!(out_down, B, x_down)
    #TODO : optimize by avoiding forming the full matrix
    out_up[:, :] .+= full(upper_offdiag * x_down)
    out_down[:, :] .+= full(lower_offdiag * x_up)
end

function (*)(x::M, right::T) where {T<:HodlrTree,M<:AbstractArray}
    _throw_error_if_incompatible_size(x, right)
    out = zeros(eltype(right), size(x, 1), size(right, 2))
    _apply_left_mul_vector_1D!(out, x, right)
    return out
end
function _apply_left_mul_vector_1D!(out, x, Hodlr::LeafHodlr)
    M = Hodlr.data
    out[:, :] .= x * M
end
function _apply_left_mul_vector_1D!(out, x, Hodlr::NodeHodlr)
    A, B, upper_offdiag, lower_offdiag = Hodlr.A, Hodlr.B, Hodlr.upper_offdiag, Hodlr.lower_offdiag
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    x_up = view(x, :, 1:nA1)
    x_down = view(x, :, nA1+1:nA1+nB1)
    out_up = view(out, :, 1:nA2)
    out_down = view(out, :, nA2+1:nA2+nB2)
    _apply_left_mul_vector_1D!(out_up, x_up, A)
    _apply_left_mul_vector_1D!(out_down, x_down, B)
    #TODO : optimize by avoiding forming the full matrix
    out_up[:, :] .+= full(x_down * lower_offdiag)
    out_down[:, :] .+= full(x_up * upper_offdiag)
end

function (*)(left::HodlrTree, right::HodlrTree)
    _throw_error_if_incompatible_size(left, right)
    return _multiply_hodlr(left, right)
end
function _multiply_hodlr(left::LeafHodlr, right::LeafHodlr)
    return LeafHodlr(left.data * right.data,combine_settings(left.settings, right.settings))
end
function _multiply_hodlr(left::NodeHodlr, right::NodeHodlr)
    new_A = left.A * right.A + left.upper_offdiag * right.lower_offdiag
    new_upper_offdiag = left.A * right.upper_offdiag + left.upper_offdiag * right.B
    new_B = left.lower_offdiag * right.upper_offdiag + left.B * right.B
    new_lower_offdiag = left.lower_offdiag * right.A + left.B * right.lower_offdiag
    return NodeHodlr(new_A, new_B, new_upper_offdiag, new_lower_offdiag)
end

function (*)(left::Number, right::NodeHodlr)
    return NodeHodlr(left*right.A, left*right.B, left*right.upper_offdiag, left*right.lower_offdiag)
end
function (*)(left::Number, right::LeafHodlr)
    return LeafHodlr(left*right.data, right.settings)
end


function _throw_error_if_incompatible_size(left, right)
    if size(left, 2) != size(right, 1)
        throw(DimensionMismatch("Non compatible dimensions for multiplication : left has size $(size(left)) and right has size $(size(right))."))
    end
end
function (-)(leaf::LeafHodlr)
    return LeafHodlr(-leaf.data, leaf.settings)
end
function (-)(tree::NodeHodlr)
    return NodeHodlr(-tree.A,-tree.B, -tree.upper_offdiag, -tree.lower_offdiag)
end
function (+)(A::LeafHodlr, B::LowRankBlock)
    return LeafHodlr(full(A) + full(B),A.settings)
end
function (+)(A::LowRankBlock, B::LeafHodlr)
    return LeafHodlr(full(A) + full(B),B.settings)
end
function (+)(A::NodeHodlr, B::LowRankBlock)
    #TODO : to improve
    return A + build_hodlr(B, A.settings)
end
function (+)(A::LowRankBlock, B::NodeHodlr)
    return build_hodlr(A, B.settings) + B
end
function (+)(left::NodeHodlr, right::NodeHodlr)
    #TODO : to improve
    return NodeHodlr(left.A + right.A, left.B + right.B, left.upper_offdiag + right.upper_offdiag, left.lower_offdiag + right.lower_offdiag)
end
function (+)(left::LeafHodlr, right::LeafHodlr)
    settings = combine_settings(left.settings, right.settings)
    return LeafHodlr(full(left) + full(right),settings)
end

function (-)(A::LeafHodlr, B::LowRankBlock)
    return LeafHodlr(full(A) - full(B),A.settings)
end
function (-)(A::LowRankBlock, B::LeafHodlr)
    return LeafHodlr(full(A) - full(B),B.settings)
end
function (-)(A::NodeHodlr, B::LowRankBlock)
    #TODO : to improve
    return A - build_hodlr(B, A.settings)
end
function (-)(A::LowRankBlock, B::NodeHodlr)
    return build_hodlr(A, B.settings) - B
end
function (-)(left::NodeHodlr, right::NodeHodlr)
    #TODO : to improve
    return NodeHodlr(left.A - right.A, left.B - right.B, left.upper_offdiag - right.upper_offdiag, left.lower_offdiag - right.lower_offdiag)
end
function (-)(left::LeafHodlr, right::LeafHodlr)
    settings = combine_settings(left.settings, right.settings)
    return LeafHodlr(full(left) - full(right),settings)
end

function is_block_upper_triangular(::LeafHodlr)
    return true
end
function is_block_upper_triangular(tree::HodlrTree)
    return iszero(tree.lower_offdiag)
end

function inv(matrix::HodlrTree)
    return inv_hodlr(matrix)
end
function inv_hodlr(tree::NodeHodlr)
    if is_block_upper_triangular(tree)
        return inv_upper_triangular_hodlr(tree)
    else
        throw(DomainError("Support only inversion of upper block triangular matrix"))
    end
end
function inv_hodlr(leaf::LeafHodlr)
    return LeafHodlr(inv(leaf.data),leaf.settings)
end
function inv_upper_triangular_hodlr(tree::NodeHodlr)
    Ainv = inv_hodlr(tree.A)
    Binv = inv_hodlr(tree.B)
    upper_off_block = -Ainv * tree.upper_offdiag * Binv
    return NodeHodlr(Ainv, Binv, upper_off_block, ZeroBlock{eltype(tree.lower_offdiag)}(size(tree.lower_offdiag)))
end
function drop_lower_block_offdiagonal(tree::NodeHodlr)
    return NodeHodlr(drop_lower_block_offdiagonal(tree.A), drop_lower_block_offdiagonal(tree.B), tree.upper_offdiag, ZeroBlock{eltype(tree)}(
        size(tree.lower_offdiag)
    ))
end
function drop_lower_block_offdiagonal(tree::LeafHodlr)
    return tree
end