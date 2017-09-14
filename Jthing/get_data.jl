push!(LOAD_PATH, @__DIR__)

using ISCLib

using HDF5, Images, NIfTI
using DataFrames,ExcelReaders, Logging, MultivariateStats, Stats
Logging.configure(level=DEBUG)

function collect_groups(labels)
    groups = [Int[] for i = 1:maximum(labels)]
    for (i,l) in enumerate(labels)
        if l != 0
            push!(groups[l], i)
        end
    end
    groups
end

δ(ind::CartesianIndex) = CartesianRange(ind-1, ind+1)

function rndsample(dfr, α::Real=0.2)
    dfr[sample(collect(1:size(dfr, 1)), Int(ceil( α * size(dfr, 1)))), :]
end

function iscmask(a)
    # load Segmentation
    mask_S = niread("$(@__DIR__)/out/S|segmentation_result.nii").raw;
    mask_V = niread("$(@__DIR__)/out/V|segmentation_result.nii").raw;
    mask_SV = niread("$(@__DIR__)/out/S_V|segmentation_result.nii").raw;

    mask_W = max(mask_S,mask_V,mask_SV);
    # mask_W = zeros(mask_S)
    # idx = (mask_S.> a) $ (mask_V.> a)
    # mask_W[idx] = mask_SV[idx]
    # mask_W
end

function glmmask(a)
    # load Segmentation
    # mask_W = niread("../../matlab/glm_res/ResMS.nii")
    mask_1 = niread("$(@__DIR__)/../../matlab/glm_res/beta_0001.nii").raw
    mask_1 = mask_1 ./ maximum(mask_1[mask_1.>0])
    mask_2 = niread("$(@__DIR__)/../../matlab/glm_res/beta_0002.nii").raw
    mask_2 = mask_2 ./ maximum(mask_2[mask_2.>0])
    mask_W = max(mask_1,mask_2)
    mask_W
end

function get_segment(a = .1, b = 0.03, pref="isc")
    print("in")
    if pref =="isc"
        mask_W = iscmask(a)
    elseif pref == "ttest" || pref == "tstat_all"
        mask_W = niread("$(@__DIR__)/out/Tstat|segmentation_result.nii").raw;
    elseif pref == "glm"
        mask_W = glmmask(a)
    end
    SVd = h5read("$(@__DIR__)/out/S_V|disps.h5", "data");

    # Average disps
    SVd = sum(SVd,2);

    med = reduce(+,[get_mean(i) for i in 1:NSUBJ])./NSUBJ
    coeff_of_var = reshape(sqrt.(SVd),SHAPE)./med
    coeff_of_var[isnan.(coeff_of_var)] = 0
    atl = niread("$(@__DIR__)/../../Harvard-Oxford\ cortical\ and\ subcortical\ structural\ atlases/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz").raw

    # Compute mask
    isc_r = copy(mask_W);
    isc_r = dilate(isc_r)
    isc_r[!((atl.>0) & (mask_W .> a) & (coeff_of_var  .>b ) & (med .> 100)) ]=0;

    ni = get_nii(1,1)
    res = NIfTI.NIVolume(ni.header, ni.extensions, isc_r)
    niwrite("$(@__DIR__)/out/$(pref)_r.nii",res)
    # get rid of noise (scanning artifacts have very high CV)
    # normed_cv = coeff_of_var./maximum(coeff_of_var)


    # get connected components
    labels = label_components(isc_r.>0)
    groups = filter(x->length(x)>40,collect_groups(labels))
    inds = get_result_inds(isc_r, groups)
    inds
end

function get_result_inds(isc_r, groups)
    mask = falses(isc_r)
    for g in groups
      mask[g] = true
    end
    inds = CartesianIndex[]
    for g in groups
        for el in g
            !mask[el] && continue
            ind = CartesianIndex(ind2sub(SHAPE,el))
            push!(inds,ind)
            for i in δ(ind)
                mask[i] = false
            end
        end
    end
    inds
end

function extract(ni, inds)
    raw = ni.raw
    out = zeros(length(inds))
    for (k,ind) in enumerate(inds)
        s = 0
        for j in δ(ind)
            s += raw[j]
        end
        out[k] = s/27
    end
    out
end

function get_params(niis, inds, num)
    """
    Gives you one point in feature space
    """
    len  = length(niis)
    stp = Int(ceil(len/num))
    frames = hcat(map(x -> extract(x, inds),niis)...)
    ms = [zeros(size(frames,1)) for i in 1:num]
    ds = [zeros(size(frames,1)) for i in 1:num]
    for i = 1:num
        ms[i] = squeeze(mean(frames[:,1+stp*(i-1): min(stp*i,len)], 2),2)
        ds[i] = squeeze(var(frames[:,1+stp*(i-1): min(stp*i,len)], 2, mean = ms[i]),2)
    end
    return vcat(ms...,ds...)
end

function get_data(inds, perc::Real = 0.2; transf = s->s[1]=='V',num = 3)
    """
    inds is a point cloud, describing a ROI, perc is the percentage of the dataset
    Gives you a sequence of points in feature space and labels for them
    (X , y)
    """
    coll = []
    for subj in shuffle(1:NSUBJ)
        data = readxl(DataFrame, joinpath(ISCLib.PATH,"logs","$subj.xlsx"), "Лист1!A1:G$(NEXP+1)");

        data[:Stimul_1] = map(transf,data[:Stimul_1])
        data[:Stimul] = map(Int,(ceil.(2*data[:Stimul])))
        data[:Response] = map(Int,(ceil.(2*data[:Response])))
        ind_end = data[[:Stimul,:Response,:Stimul_1]][2:end-2,:]
        push!(coll,(subj,ind_end))
    end
    count = 0
    X = Vector{Float32}[]
    y = []
    for col in coll
        subj = col[1]
        dfr = col[2]
        debug(subj)
        idxs = Array(rndsample(dfr,perc))
        for i in 1:size(idxs,1)
            row = view(idxs, i, :)
            if row[2]- row[1] < num*2
                continue
            else
                params = get_params(get_niis(subj,row[1]:row[2]),inds, num)
                push!(X,params)
                push!(y,row[3])
            end
        end
    end
    X,y
end

function extract_data(;a = .4, b = 0.02, rnd = false, pref = "", num = 3,)
    if rnd
        inds = [CartesianIndex((rand(10:80),rand(10:80),rand(10:80))) for i in 1:500]
    else
        inds = get_segment(a, b, pref);
    end

    X,y = get_data(inds, 0.99, transf = x->x, num = num);
    X = hcat(X...)
    y = Vector{typeof(y[1])}(y)
    h5open("out/$pref|res_x.h5", "w") do file
        write(file, "data", X)
    end
    h5open("out/$pref|res_y.h5", "w") do file
        write(file,"data", y)
    end
    X,y
end

# extract_data(a = 0.16, b = 0.03, pref = "isc", num=3)
extract_data(a = 40, b = 0.03, pref = "tstat_all", num=3)
# extract_data(a = 0.16, b = 0.03, pref = "glm", num=3)

# extract_data(rnd = true, pref = "rnd")
