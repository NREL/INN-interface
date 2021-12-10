import matplotlib.pyplot as plt
import numpy as np

a = {'wt.wt_init.blade.opt_var.chord_opt': np.array([5.2       , 5.75402153, 4.16137546, 3.46440473, 2.20239731,
       0.5       ]),
 'wt.wt_init.blade.opt_var.spar_cap_ss_opt': np.array([0.00011402, 0.07375611, 0.11304067, 0.09347514, 0.03782193,
       0.001     ]),
 'wt.wt_init.inn_af.L_D_opt': np.array([2.85714286e-04, 4.47264153e+01, 1.08528986e+02, 7.01111029e+01,
       9.97543463e+01, 1.01467868e+02]),
 'wt.wt_init.inn_af.c_d_opt': np.array([0.35      , 0.03612836, 0.01553854, 0.01507587, 0.01289341,
       0.01639005]),
 'wt.wt_init.inn_af.r_thick_opt': np.array([1.        , 0.40976896, 0.33172176, 0.25530248, 0.20303135,
       0.21822208]),
 'wt.wt_init.inn_af.stall_margin_opt': np.array([-0.85824848,  0.09358083,  0.00786694,  0.036415  ,  0.06602035,
        0.26726918]),
 'wt.wt_init.inn_af.z': np.array([0.42034738, 0.75760499, 0.36806643])}
 
b = {'wt.wt_init.blade.opt_var.chord_opt': np.array([5.2       , 5.75402153, 4.88661534, 4.5957393 , 1.63125629,
    0.5       ]),
'wt.wt_init.blade.opt_var.spar_cap_ss_opt': np.array([0.0001142 , 0.07206098, 0.10442782, 0.0966301 , 0.03463771,
    0.001     ]),
'wt.wt_init.inn_af.L_D_opt': np.array([2.85714286e-04, 4.47264153e+01, 1.10111255e+02, 7.11198524e+01,
    9.85227947e+01, 1.03297019e+02]),
'wt.wt_init.inn_af.c_d_opt': np.array([0.35      , 0.03612836, 0.015643  , 0.01543486, 0.01261582,
    0.01653843]),
'wt.wt_init.inn_af.r_thick_opt': np.array([1.        , 0.40976896, 0.33321325, 0.26766725, 0.20391658,
    0.21596029]),
'wt.wt_init.inn_af.stall_margin_opt': np.array([-0.85824848,  0.09358083,  0.0332434 ,  0.01070734,  0.04228639,
     0.2748356 ]),
'wt.wt_init.inn_af.z': np.array([0.64777319, 0.69428036, 0.30559425])}

c = {'wt.wt_init.blade.opt_var.chord_opt': np.array([5.2       , 5.75402153, 5.37940085, 4.88545352, 2.68458257,
       0.5       ]),
 'wt.wt_init.blade.opt_var.spar_cap_ss_opt': np.array([1.11965090e-04, 7.33938935e-02, 1.17664615e-01, 1.02708079e-01,
       3.65434233e-02, 1.00000000e-03]),
 'wt.wt_init.inn_af.L_D_opt': np.array([2.85714286e-04, 4.47264153e+01, 1.21803832e+02, 6.84848849e+01,
       9.21360373e+01, 1.00358233e+02]),
 'wt.wt_init.inn_af.c_d_opt': np.array([0.35      , 0.03612836, 0.01516675, 0.01699783, 0.01126729,
       0.01786264]),
 'wt.wt_init.inn_af.r_thick_opt': np.array([1.        , 0.40976896, 0.30708548, 0.24980068, 0.23659905,
       0.1924176 ]),
 'wt.wt_init.inn_af.stall_margin_opt': np.array([-0.85824848,  0.09358083,  0.08430232, -0.02091844,  0.04226919,
        0.27938159]),
 'wt.wt_init.inn_af.z': np.array([0.41918037, 0.76084641, 0.158474  ])}

all_iters = [a, b]

fig, axarr = plt.subplots(7, 1, figsize=(4, 12))
for jdx, it in enumerate(all_iters):
    for idx, key in enumerate(it):
        dat = it[key]
        lins = np.linspace(0., 1., len(dat))
        axarr[idx].plot(lins, dat, label=f'{jdx}')
        axarr[idx].set_ylabel(key.split('.')[-1])
plt.legend()
        
plt.tight_layout()
plt.savefig('compare_iterations.png')