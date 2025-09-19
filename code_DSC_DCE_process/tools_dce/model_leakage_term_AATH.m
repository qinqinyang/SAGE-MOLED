function Ct = model_leakage_term_AATH(time, Ktrans, ve, Tc)

Ct = Ktrans*exp(-(Ktrans/ve)*(time-Tc));

end