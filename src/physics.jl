export theq_lesser_time_kernel
export thermal_kernel
function theq_lesser_time_kernel(T,bs,τ; kb = 1., ħ = 1.)
    f_reg(t,tp) = diagm([ thermal_kernel(t-tp,T,τ, kb = kb, ħ = ħ) for i = 1:bs])
    f_δ(t) = diagm([-1. /ħ for i = 1:bs])
    return f_δ, f_reg
end

function thermal_kernel(t,T,τ; kb = 1., ħ = 1.)
    α = 1/(2*kb*T)
    r = (τ *((4im*t)/(t^2 + τ^2*ħ^2) + 
    (polygamma(0,(-1im*t + τ *ħ)/(4. *α *ħ)) - 
       polygamma(0,(1im*t + τ *ħ)/(4. *α *ħ)) - 
       polygamma(0,(-1im*t + 2*α *ħ + τ *ħ)/(4. *α *ħ)) + 
       polygamma(0,(1im*t + 2*α *ħ + τ *ħ)/(4. *α *ħ)))/(α *ħ)))/(8. *π)
    return r*2/τ
end

function thermal_kernel(t,β)
    t_min = 1e-64
    if abs(t) > 1e-64
        return -1im/β * csch(π*t/β)
    else 
        return Complex{typeof(t)}(0)
    end
end


function pauli(k::Int)
    if k == 1
        return @SMatrix[0 1; 1 0]
    elseif k == 2
        return @SMatrix[0 -1im; 1im 0]
    elseif k == 3
        return  @SMatrix[1 0; 0 -1]
    else
        return @SMatrix[1 0; 0 1]
    end
end
#=
function RAK_rotation()
    (1 .- 1im*pauli(2))/sqrt(2)
end
function pauliRAK(k)
    RAK_rotation()*pauli(k)*RAK_rotation()'
end
=#