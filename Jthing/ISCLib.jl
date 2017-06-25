module ISCLib

using Stats, NIfTI, Logging, ExcelReaders, DataFrames


include("consts.jl")

export NSUBJ, SHAPE, SIZE, get_nii,
      get_niis, cov_n!, LEN,
      get_mean, triangular_index, ISC_res,
      get_indexes, NEXP

"""
# indexes for the end of response for each question
# i - Subject_id
"""

function get_indexes(subj::Int, t::Array{String,1} = ["S1","S2"], rb::Int64=1)
    data = readxl(DataFrame,joinpath(PATH,"logs","$subj.xlsx"), "Лист1!A1:G$(NEXP+1)");
    data = data[reduce((x,y)-> x .| y,(Array(data[:Stimul_1].== w) for w in t)),:]
    ind_end = data[:Response]
    # map(Int,(ceil(hcat(ind_end-minimum(data[Symbol("Stim dur")]),ind_end))*2)) #ceil returns float64
    map(Int,(ceil(hcat(ind_end-6,ind_end+2))*2))[1:end-1-rb,:]
end

function get_nii(subj::Int, ind::Int)
    fnames = readdir(PATH_Fmt(subj))[2:end] # first elem is  mean
    return niread(joinpath(PATH_Fmt(subj),fnames[ind]),mmap= true)
end

function get_niis(subj::Int, rng::AbstractArray=1:LEN)
    fnames = readdir(PATH_Fmt(subj))[2:end] # first elem is  mean
    return [niread(joinpath(PATH_Fmt(subj),fnames[ind]),mmap = true) for ind in rng]
end

@inline function rema!{T<: Real}(shadow::Array{T}, variable::Array{T}, decay = 0.9)
    shadow .-= (1 - decay) * (shadow - variable)
end

function get_mean(subj::Int)
  debug("mean",subj)
  if isfile(joinpath(PATH_Fmt(subj),"mean.nii"))
    return niread(joinpath(PATH_Fmt(subj),"mean.nii"))
  else
    seq = get_niis(subj,1:LEN)
    m = mean(seq)
    ni = NIfTI.NIVolume(seq[1].header, seq[1].extensions, m)
    niwrite(joinpath(PATH_Fmt(subj),"mean.nii"), ni)
  end
  m
end

function triangular_index(i::Int,j::Int)
  Int(round(j * (j - 3) / 2 + i + 1))
end

function reverse_tri_index(k::Int)
  j = round(floor(-.5 + .5 * sqrt(1 + 8 * (k - 1))) + 2);
  i = round(j * (3 - j) / 2 + k - 1);
  Int(i),Int(j)
end

function cov_n!{T<:Real}(covs::Array{T,2},ds::Array{T,2},seq::AbstractArray,means::AbstractArray,lim::Int)
  info("cov_n called")
  srol = [seq[i][1].raw - means[i] for i in 1:NSUBJ ]
  for i in 2:lim
      @simd for subj in 1:NSUBJ
          rema!(srol[subj],seq[subj][i].raw - means[subj])
          # in-place Фильтр Брауна по 1 аргументу ( x = δ x + (1-δ) χ, χ - new x )
          @inbounds ds[:,subj] .+= reshape(srol[subj].^2, SIZE)
      end
      @simd for k in 2:NSUBJ for j in 1:k-1
              # info(i,j,k,"  ",triangular_index(j,k))
              @inbounds covs[:,triangular_index(j,k)] .+= reshape(srol[j] .* srol[k], SIZE)
      end end
  end
end

function ISC_res{T<:Real}(covs::Array{T,2},ds::Array{T,2})
    res = zeros(SIZE)
    for i in 1:SIZE
      for k in 2:NSUBJ for j in 1:k-1
          res[i] += covs[i,triangular_index(j,k)]/sqrt(ds[i,j]*ds[i,k])
      end end
      res[i] /= (NSUBJ*(NSUBJ-1))/2
    end
    res
end

end
