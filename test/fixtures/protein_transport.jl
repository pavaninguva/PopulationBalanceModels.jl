function protein_transport_fixture()
    coefficient=[-22.2 -3.35 1.1 2.2;
                 .1 .03 -.2 .4;
                 -.02 .01 .05 -.1;
                 0. .002 .03 .02]
    low=Matrix(Diagonal([.1,.05,.4,.3]));low[2,1]=.03;low[4,2]=.12
    high=Matrix(Diagonal([.2,.07,.3,.25]));high[3,1]=.1;high[4,2]=-.08
    (;mixture_means=[-1.,1.5],mixture_scales=[.2,1.],mixture_weights=[.2,.8],
      polynomial_coefficients=coefficient,low_cholesky=low,high_cholesky=high,
      low_mean=[.02,-.01,.03,-.02],high_mean=[-.01,.02,-.02,.01],
      parent_center=70.,parent_scale=30.,blend_center=37.,blend_width=3.)
end
