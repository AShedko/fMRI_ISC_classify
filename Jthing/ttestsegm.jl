using HypothesisTests, HDF5, NIfTI
using Distributions
push!(LOAD_PATH, ".")
using ISCLib

function TTest(x::AbstractVector, y::AbstractVector, μ0=0)
    nx, ny = length(x), length(y)
    m1, m2 = sum(x)/nx,sum(y)/ny
    xbar = m1-m2
    varx, vary = reduce((a,b)-> a+(b-m1).^2, x)/(nx-1), reduce((a,b)-> a+(b-m2).^2, y)/(ny-1)
    stderr = sqrt.(varx/nx + vary/ny)
    t = (xbar-μ0)./stderr
    df = (varx ./ nx + vary ./ ny).^2 ./ ((varx / nx).^2 ./ (nx - 1) + (vary ./ ny).^2 / (ny - 1))
    (nx, ny, xbar, df, stderr, t)
end

function segment(subj::Int)
    sind = get_indexes(subj,["S1","S2"]);
    vind = get_indexes(subj,["V1","V2","V3","V4"]);
    S_seq = vcat([get_niis(subj,sind[i,1]:sind[i,2]) for i =1:size(sind,1)]...);
    V_seq = vcat([get_niis(subj,vind[i,1]:vind[i,2]) for i =1:size(vind,1)]...);
    (nx, ny, xbar, df, stderr, t) = TTest(S_seq,V_seq);

    df = (x->isnan(x)?0:x).(df)
    t = (x->isnan(x)?0:x).(t)
    pvals = zeros(df)
    for (i,(df_el,t_el)) = enumerate(zip(df,t))
        try
            pvals[i] = pvalue(TDist(df_el),t_el)
        catch
            pvals[i] = 1
        end
    end

    pvals
    logpvals = -log10.(pvals);
    logpvals[logpvals < 5] = 0
    # logpvals[logpvals > 20] = 20
    ni = get_nii(1,1)
    res = NIfTI.NIVolume(ni.header, ni.extensions,logpvals) # Cheap, as reshape returns a view
    niwrite("out/$pref|segmentation_result.nii",res)
    logpvals
end
