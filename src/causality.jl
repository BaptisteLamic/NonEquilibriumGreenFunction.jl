abstract type AbstractCausality end
struct Retarded <: AbstractCausality end
struct Acausal <: AbstractCausality end
struct Advanced <: AbstractCausality end
struct Instantaneous <: AbstractCausality end

causality_of_sum(::C, ::C) where {C<:AbstractCausality} = C()
causality_of_sum(::AbstractCausality, ::AbstractCausality) = Acausal()
causality_of_sum(left::AbstractCausality, ::Instantaneous) = left
causality_of_sum(::Instantaneous, right::AbstractCausality) = right
causality_of_sum(::Instantaneous, ::Instantaneous) = Instantaneous

causality_of_prod(::Retarded, ::Retarded) = Retarded()
causality_of_prod(::Advanced, ::Advanced) = Advanced()
causality_of_prod(::Retarded, ::Acausal) = Acausal()
causality_of_prod(::Acausal, ::Advanced) = Acausal()
causality_of_prod(::Acausal, ::Acausal) = Acausal()
causality_of_prod(::Instantaneous, ::Instantaneous) = Instantaneous()
causality_of_prod(::Instantaneous, ::T) where {T<:AbstractCausality} = T()
causality_of_prod(::T, ::Instantaneous) where {T<:AbstractCausality} = T()