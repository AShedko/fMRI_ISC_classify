module ISCLib

using Formatting, Stats, NIfTI
# const PATH = "/home/ashedko/Projects/UIR/fmri/classify/CogVis_4trofimov/{1:d}/fmri/"
# const PATH = "/run/media/ashedko/5cdd1287-7dba-4138-8a1b-91148f9f8ab5/ashedko/CogVis_4trofimov/{1:d}/fmri/"
# const PATH = "/run/media/ashedko/TOSHIBA EXT/UNC/{1:d}/"
const PATH = "/run/media/ashedko/5cdd1287-7dba-4138-8a1b-91148f9f8ab5/ashedko/UNC/{1:d}/"
const PATH_Fmt = x -> format( PATH , x)
const NSUBJ = 29 # 7 in test runs
const SIZEX = 91
const SIZEY = 109
const SIZEZ = 91
const SHAPE = (SIZEX,SIZEY,SIZEZ)
const SIZE = SIZEX * SIZEY * SIZEZ
const LEN = 3620
export NSUBJ, SHAPE, SIZE, get_niis, cor_n, cov_n, rema!, LEN

function get_nii(subj::Int, rng::AbstractArray)
    return niread(joinpath(PATH_Fmt(subj),FNAMES[subj][ind]),mmap= true)
end

function get_niis(subj::Int, rng::AbstractArray=1:LEN)
    fnames = readdir(PATH_Fmt(subj))
    return [niread(joinpath(PATH_Fmt(subj),fnames[ind]),mmap= true) for ind in rng]
end

function rema!(shadow::Array, variable::Array,decay::Float64= 0.3)
    shadow -= (1 - decay) * (shadow - variable)
end

function prep(subj::Int64)
  seq = get_niis(subj,1:LEN)
  m = mean(seq)
  niwrite(joinpath(PATH_Fmt(subj),"mean.nii"),m)
  m
end

function cov_n(seq::AbstractArray,means::AbstractArray,indexes::AbstractArray=1:LEN)
  srol = [seq[i][indexes[1]].raw - means[i] for i in 1:length(seq)]
  n_sub = length(seq)
  covs = zeros(SIZE,n_sub, n_sub) # ковариационные матрицы
  ds = [zeros(SIZE) for i in 1:n_sub]
  @sync @parallel for i in indexes[2:end]
      @simd for ind in 1:n_sub
          srol[ind] = rema!(srol[ind],seq[ind][i].raw - means[ind])
          # in-place Фильтр Брауна по 1 аргументу ( x = (1-δ)x + χ, χ - new x )
          @inbounds ds[ind] += reshape(srol[ind] .^2, SIZE)
      end
      @simd for k in 1:n_sub
          for j in k+1:n_sub
              covs[:,k,j] += reshape(srol[j] .* srol[k], SIZE)
          end
      end
  end
  covs,ds
end

function cor_n(means::AbstractArray,indexes::AbstractArray=1:LEN)
    covs,ds = cov_n(seq,means,indexes)
    corrmats = zeros(covs)
    n_sub = length(seq)
    for i in 1:SIZE
        corrmats[i,:,:] = eye(n_sub)
        for j in 1:(n_sub-1)
            for k in (j+1):n_sub
                corrmats[i,j,k] = covs[i,j,k]/sqrt(ds[j][i]*ds[k][i])
                corrmats[i,k,j] = corrmats[i,j,k]
            end
        end
    end
    corrmats,ds
end

end
