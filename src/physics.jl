
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

