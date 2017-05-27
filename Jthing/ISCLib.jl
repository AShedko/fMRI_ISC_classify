module ISCLib

using Stats, NIfTI, Logging, ExcelReaders, DataFrames
@Logging.configure(level=DEBUG)

const PATH = "/run/media/ashedko/5cdd1287-7dba-4138-8a1b-91148f9f8ab5/ashedko/UNC/"
const PATH_Fmt = x -> PATH * string(x) * "/"
const NSUBJ = 29 # 7 in test runs
const SIZEX = 91
const SIZEY = 109
const SIZEZ = 91
const SHAPE = (SIZEX,SIZEY,SIZEZ)
const SIZE = SIZEX * SIZEY * SIZEZ
const LEN = 3620
const NEXP = 91
export NSUBJ, SHAPE, SIZE, get_nii,
      get_niis, cov_n!, rema!, LEN,
      prep, triangular_index, ISC_res,
      get_indexes, NEXP

macro s_str(s)
  Expr(:quote, symbol(s))
end

"""
# indexes for the end of response for each question
# i - Subject_id
"""
function get_indexes(i::Int64, t::Array{String,1} = ["S1","S2","S3","S4"])
    data = readxl(DataFrame,joinpath(PATH,"logs","$i.xlsx"), "Лист1!A1:G$(NEXP+1)");
    data = data[reduce(|,data[:Stimul_1] .== w for w in t),:]
    ind_end = data[:Response]
    # map(Int,(ceil(hcat(ind_end-minimum(data[Symbol("Stim dur")]),ind_end))*2)) #ceil returns float64
    map(Int,(ceil(hcat(ind_end-6,ind_end+2))*2))[1:end-1,:]
end

function get_nii(subj::Int, ind::Int)
    fnames = readdir(PATH_Fmt(subj))
    return niread(joinpath(PATH_Fmt(subj),fnames[ind]),mmap= true)
end

function get_niis(subj::Int, rng::AbstractArray=1:LEN)
    fnames = readdir(PATH_Fmt(subj))
    return [niread(joinpath(PATH_Fmt(subj),fnames[ind]),mmap= true) for ind in rng]
end

@inline function rema!(shadow::Array, variable::Array,decay::Float64= 0.3)
    shadow .-= (1 - decay) * (shadow - variable)
end

function prep(subj::Int64)
  @debug("mean",subj)
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
  i = round(j * (3 - J) / 2 + k - 1);
  Int(i),Int(j)
end

function cov_n!(covs::Array{Float64,2},ds::Array{Float64,2},seq::AbstractArray,means::AbstractArray,lim::Int64)
  info("cov_n called")
  srol = [seq[i][1].raw - means[i] for i in 1:NSUBJ  ]
  for i in 2:lim
      @simd for ind in 1:NSUBJ
          srol[ind] .= rema!(srol[ind],seq[ind][i].raw - means[ind])
          # in-place Фильтр Брауна по 1 аргументу ( x = (1-δ)x + χ, χ - new x )
          @inbounds ds[:,ind] .+= reshape(srol[ind] .^2, SIZE)
      end
      @simd for k in 2:NSUBJ for j in 1:k-1
              # info(i,j,k,"  ",triangular_index(j,k))
              @inbounds covs[:,triangular_index(j,k)] .+= reshape(srol[j] .* srol[k], SIZE)
      end end
  end
end

function ISC_res(covs::Array{Float64,2},ds::Array{Float64,2})
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
