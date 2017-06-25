module Temp
  function rema_!(shadow::Float64,variable::Float64,decay::Float64= 0.3)
      shadow -= (1 - decay) * (shadow - variable)
  end

  function smooth(x,decay=0.9)
    res = copy(x)
    st = x[1]
    res[1] = st
    i = 2
    for el in x[2:end]
      st = rema_!(st,el,decay)
      res[i] = st
      i+=1
    end
    res
  end

end
