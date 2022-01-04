import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots
from time import time

import hashlib
import dill
from pathlib import Path


def load_cases(case_names=None, just_print_names=False):

    Path("saved_results").mkdir(parents=True, exist_ok=True)
    
    optimization_logs = []

    # Simply gather all of the sql files
    run_dir = os.path.dirname(os.path.realpath(__file__))
    for subdir, dirs, files in os.walk(run_dir):
        for file in files:
            if 'sql' in file:
                optimization_logs.append(os.path.join(subdir, file))
    
    if case_names is not None:
        case_filenames = []
        for case_name in case_names:
            for log in optimization_logs:
                if case_name in log:
                    case_filenames.append(log)
                    
        optimization_logs = case_filenames
        
    s = time()
    all_data = []
    for idx, log in enumerate(optimization_logs):
        print(f"Loading case {idx} / {len(optimization_logs)}")

        readable_hash = hashlib.md5(str.encode(log)).hexdigest()
     
        filename = os.path.join('saved_results', f'{readable_hash}.pkl')
        
        if just_print_names:
            print(log, readable_hash)
            continue
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = dill.load(f)
        else:
            data = {}
            cr = om.CaseReader(log)
            cases = cr.get_cases()
            
            for case in cases:
                for key in case.outputs.keys():
                    if key not in data.keys():
                        data[key] = []
                        # print(key)
                    data[key].append(case.outputs[key])
            
            for key in data.keys():
                data[key] = np.array(data[key])
                
            # Manually compute t/c and add
            coords = data['blade.run_inn_af.coord_xy_interp']
            lower = coords[:, :, :100, 1]
            upper = coords[:, :, 100:, 1][::-1]
            diff = np.max(upper - lower, axis=2)
            data['t/c'] = diff
                
            with open(filename, 'wb') as f:
                dill.dump(data, f)
            
        all_data.append(data)
        
    return all_data, optimization_logs
    
    
if __name__ == "__main__":
    load_cases(['00'], just_print_names=False)
    
    
# costs.tower_mass_cost_coeff
# rotorse.ccblade.L_n_opt
# towerse.monopile_cost
# rotorse.re.y_tc
# rotorse.xu_te
# rotorse.rs.x_az
# rotorse.rs.tip_pos.tip_deflection
# rotorse.rp.powercurve.P_spline
# costs.wake_loss_factor
# rotorse.rs.tot_loads_gust.Pz_af
# blade.internal_structure_2d_fem.web_end_nd_yaml
# rotorse.rs.constr.constr_flap_f_margin
# costs.lss_mass_cost_coeff
# rotorse.ccblade.LiftF
# rotorse.ccblade.Px_af
# rotorse.rs.aero_gust.loads_Pz
# blade.high_level_blade_props.blade_ref_axis
# rotorse.ccblade.Q
# rotorse.ccblade.D_n_opt
# rotorse.rc.total_cost_building
# rotorse.ccblade.cl_n_opt
# costs.hvac_mass_cost_coeff
# rotorse.rs.strains.axial_maxc_teU_load2stress
# inn_af.stall_margin_opt
# rotorse.rc.sect_perimeter
# rotorse.rc.blade_variable_cost
# rotorse.rs.frame.EI11
# rotorse.wakerotation
# rotorse.re.precomp.mass_all_blades
# bos.port_cost_per_month
# rotorse.rp.AEP
# towerse.structural_cost
# rotorse.rp.powercurve.tang_induct_regII
# rotorse.rs.constr.constr_max_strainU_te
# tcc.brake_cost
# blade.internal_structure_2d_fem.definition_layer
# rotorse.tiploss
# rotorse.rc.total_labor_overhead
# blade.pa.chord_param
# rotorse.re.x_cg
# blade.fatigue.teL_wohlerexp
# costs.cover_mass_cost_coeff
# rotorse.rp.powercurve.Cp_aero
# rotorse.A
# monopile.layer_thickness
# tower.layer_thickness
# inn_af.s_opt_stall_margin
# blade.internal_structure_2d_fem.web_start_nd
# rotorse.rp.powercurve.V
# inn_af.L_D_opt
# rotorse.rs.frame.freq_distance
# blade.compute_reynolds.Re
# rotorse.usecd
# rotorse.y_ec
# rotorse.rp.gust.V_gust
# blade.interp_airfoils.cl_interp
# rotorse.wt_class.V_mean
# rotorse.rs.z_az
# rotorse.rc.total_non_gating_ct
# blade.fatigue.sparL_wohlerA
# rotorse.rp.powercurve.pitch
# costs.bearing_mass_cost_coeff
# tcc.hub_system_cost
# rotorse.rs.aero_gust.loads_Px
# tcc.transformer_cost
# blade.internal_structure_2d_fem.web_rotation_yaml
# rotorse.xl_spar
# rotorse.rc.mat_cost_scrap
# inn_af.z
# rotorse.GJ
# rotorse.rp.powercurve.aoa_regII
# blade.internal_structure_2d_fem.layer_side
# rotorse.rp.powercurve.Ct_aero
# blade.internal_structure_2d_fem.layer_end_nd
# rotorse.rs.aero_hub_loads.Mb
# tcc.blade_cost
# rotorse.rc.total_cost_labor
# rotorse.rc.mat_mass
# blade.outer_shape_bem.twist_yaml
# rotorse.rc.total_cost_equipment
# control.max_pitch_rate
# rotorse.rs.brs.d_r
# blade.run_inn_af.cl_interp
# blade.internal_structure_2d_fem.joint_mass
# rotorse.rc.total_skin_mold_gating_ct
# blade.internal_structure_2d_fem.layer_start_nd
# costs.bedplate_mass_cost_coeff
# tcons.tip_deflection_ratio
# rotorse.rs.aero_hub_loads.CMhub
# rotorse.wt_class.V_extreme1
# rotorse.re.precomp.blade_mass
# rotorse.ccblade.T
# rotorse.rs.constr.constr_max_strainL_spar
# drivese.bedplate_mat_cost
# blade.outer_shape_bem.s
# blade.opt_var.s_opt_te_ps
# blade.fatigue.teL_sigma_ult
# rotorse.rs.frame.root_F
# control.rated_pitch
# costs.spinner_mass_cost_coeff
# blade.run_inn_af.aoa_inn
# rotorse.rp.powercurve.rated_T
# rotorse.rs.frame.dx
# tcc.turbine_cost_kW
# blade.internal_structure_2d_fem.layer_web
# tcc.generator_cost
# rotorse.rs.strains.strainU_spar
# rotorse.rp.powercurve.P
# blade.internal_structure_2d_fem.web_end_nd
# blade.opt_var.te_ss_opt
# rotorse.re.sc_ps_mats
# blade.high_level_blade_props.prebendTip
# rotorse.yu_spar
# blade.outer_shape_bem.twist
# rotorse.rs.frame.edge_mode_shapes
# rotorse.stall_check.no_stall_constraint
# rotorse.yl_te
# rotorse.rs.aero_hub_loads.P
# rotorse.re.x_tc
# blade.internal_structure_2d_fem.layer_offset_y_pa
# blade.fatigue.teU_wohlerexp
# costs.turbine_number
# rotorse.x_ec
# blade.internal_structure_2d_fem.layer_rotation_yaml
# blade.internal_structure_2d_fem.layer_width
# rotorse.xl_te
# blade.internal_structure_2d_fem.sigma_max
# tcc.turbine_cost
# blade.internal_structure_2d_fem.layer_rotation
# blade.interp_airfoils.r_thick_interp
# drivese.lss_cost
# drivese.pitch_mass
# rotorse.rp.powercurve.T
# tcc.hub_cost
# costs.blade_mass_cost_coeff
# rotorse.rc.total_cost_tooling
# tcc.converter_cost
# blade.compute_coord_xy_dim.coord_xy_dim
# rotorse.rc.total_labor_hours
# blade.fatigue.teL_wohlerA
# drivese.hub_system_cost
# rotorse.re.y_sc
# rotorse.rs.aero_hub_loads.CFhub
# blade.internal_structure_2d_fem.layer_start_nd_yaml
# blade.ps.layer_thickness_param
# rotorse.rc.mat_volume
# costs.elec_connec_machine_rating_cost_coeff
# blade.interp_airfoils.cm_interp
# drivese.hub_cost
# rotorse.rc.total_metallic_parts_cost
# costs.gearbox_mass_cost_coeff
# rotorse.re.precomp.blade_moment_of_inertia
# tcc.main_bearing_cost
# blade.opt_var.te_ps_opt
# rotorse.rs.strains.strainU_te
# blade.pa.max_chord_constr
# blade.internal_structure_2d_fem.index_layer_start
# tcons.blade_tip_tower_clearance
# rotorse.rs.strains.axial_root_sparL_load2stress
# tcc.pitch_system_cost
# rotorse.ccblade.M
# rotorse.re.Tw_iner
# inn_af.s_opt_r_thick
# blade.internal_structure_2d_fem.layer_end_nd_yaml
# rotorse.rs.frame.EI22
# tcc.platforms_cost
# bos.site_assessment_plan_cost
# towerse.unit_cost
# blade.internal_structure_2d_fem.web_rotation
# tcc.gearbox_cost
# rotorse.rs.frame.all_mode_shapes
# rotorse.rp.powercurve.Q
# rotorse.rs.strains.axial_root_sparU_load2stress
# rotorse.rs.aero_gust.loads_r
# rotorse.EIyy
# blade.fatigue.sparU_sigma_ult
# tcc.cover_cost
# blade.fatigue.teU_wohlerA
# blade.high_level_blade_props.r_blade
# rotorse.rs.frame.root_M
# rotorse.stall_check.stall_angle_along_span
# rotorse.nSector
# blade.run_inn_af.coord_xy_interp
# rotorse.yu_te
# blade.internal_structure_2d_fem.index_layer_end
# rotorse.rp.powercurve.rated_Omega
# rotorse.ccblade.Px_b
# blade.opt_var.s_opt_spar_cap_ss
# blade.outer_shape_bem.s_default
# rotorse.re.precomp.I_all_blades
# costs.pitch_system_mass_cost_coeff
# rotorse.rp.powercurve.P_aero
# blade.high_level_blade_props.prebend
# tcc.hvac_cost
# blade.fatigue.sparU_wohlerexp
# rotorse.re.x_sc
# blade.outer_shape_bem.af_position
# rotorse.rc.total_maintenance_cost
# tcc.spinner_cost
# blade.interp_airfoils.cd_interp
# blade.internal_structure_2d_fem.layer_thickness
# rotorse.re.y_cg
# blade.internal_structure_2d_fem.web_offset_y_pa
# drivese.hss_cost
# rotorse.theta
# rotorse.rhoJ
# blade.fatigue.teU_sigma_ult
# rotorse.EIxx
# towerse.unit_cost_full
# tcc.yaw_system_cost
# inn_af.s_opt_c_d
# rotorse.ccblade.alpha
# rotorse.ccblade.ap
# rotorse.yl_spar
# drivese.spinner_cost
# blade.pa.twist_param
# rotorse.rs.3d_curv
# blade.internal_structure_2d_fem.web_start_nd_yaml
# rotorse.rs.tot_loads_gust.Py_af
# costs.painting_rate
# tcc.bedplate_cost
# blade.high_level_blade_props.blade_length
# blade.outer_shape_bem.pitch_axis
# rotorse.rp.powercurve.Omega_spline
# rotorse.rc.total_consumable_cost_w_waste
# blade.outer_shape_bem.ref_axis
# rotorse.hubloss
# rotorse.rs.frame.flap_mode_shapes
# tcc.controls_cost
# bos.design_install_plan_cost
# rotorse.rp.powercurve.L_D
# rotorse.rs.frame.edge_mode_freqs
# configuration.n_blades
# rotorse.ccblade.a
# costs.labor_rate
# costs.hss_mass_cost_coeff
# blade.opt_var.twist_opt
# blade.outer_shape_bem.chord_yaml
# rotorse.rs.frame.F3
# rotorse.ccblade.cd
# rotorse.rp.powercurve.ax_induct_regII
# rotorse.rc.total_blade_cost
# rotorse.rs.strains.strainL_spar
# blade.internal_structure_2d_fem.layer_width_yaml
# rotorse.re.sc_ss_mats
# tcc.elec_cost
# blade.internal_structure_2d_fem.layer_midpoint_nd
# rotorse.rs.tot_loads_gust.Px_af
# towerse.tower_cost
# costs.hub_mass_cost_coeff
# tcc.nacelle_cost
# towerse.cm.cost
# bos.boem_review_cost
# rotorse.rp.powercurve.Cp_regII
# airfoils.coord_xy
# costs.transformer_mass_cost_coeff
# blade.fatigue.sparL_wohlerexp
# rotorse.ccblade.cd_n_opt
# rotorse.rp.powercurve.rated_pitch
# rotorse.rs.constr.constr_max_strainU_spar
# blade.opt_var.chord_opt
# blade.opt_var.af_position
# rotorse.EA
# costs.yaw_mass_cost_coeff
# drivese.pitch_cost
# rotorse.rp.powercurve.cl_regII
# rotorse.rs.brs.ratio
# blade.high_level_blade_props.rotor_radius
# rotorse.ccblade.DragF
# costs.bos_per_kW
# blade.opt_var.s_opt_chord
# rotorse.rs.curvature.s
# rotorse.re.precomp.z
# rotorse.rs.strains.strainL_te
# rotorse.rp.powercurve.rated_Q
# blade.internal_structure_2d_fem.joint_position
# rotorse.rp.powercurve.V_R25
# rotorse.rs.aero_hub_loads.CP
# rotorse.rs.frame.M1
# rotorse.rp.powercurve.Cq_aero
# financese.lcoe
# blade.opt_var.s_opt_twist
# blade.run_inn_af.cd_interp
# rotorse.xu_spar
# rotorse.rc.cost_capital
# tcc.hss_cost
# rotorse.rp.powercurve.Cm_aero
# rotorse.ccblade.P
# blade.internal_structure_2d_fem.joint_cost
# blade.internal_structure_2d_fem.definition_web
# rotorse.ccblade.Py_af
# rotorse.re.te_ps_mats
# blade.interp_airfoils.ac_interp
# rotorse.rs.frame.flap_mode_freqs
# blade.fatigue.sparL_sigma_ult
# costs.opex_per_kW
# blade.outer_shape_bem.ref_axis_yaml
# inn_af.r_thick_opt
# rotorse.ccblade.local_airfoil_velocities
# rotorse.rp.powercurve.M
# blade.opt_var.s_opt_spar_cap_ps
# rotorse.ccblade.Pz_af
# hub.pitch_system_scaling_factor
# rotorse.EIxy
# rotorse.re.precomp.edge_iner
# rotorse.rp.powercurve.Cp
# drivese.spinner_mat_cost
# rotorse.rc.layer_volume
# rotorse.rp.powercurve.rated_V
# rotorse.rs.strains.axial_maxc_teL_load2stress
# rotorse.rs.frame.M2
# costs.platforms_mass_cost_coeff
# blade.opt_var.spar_cap_ss_opt
# rotorse.rs.aero_hub_loads.Fhub
# blade.outer_shape_bem.chord
# drivese.hub_mat_cost
# bos.construction_operations_plan_cost
# rotorse.rc.total_blade_mat_cost_w_waste
# blade.internal_structure_2d_fem.web_offset_y_pa_yaml
# tcc.lss_cost
# inn_af.c_d_opt
# rotorse.rp.powercurve.rated_mech
# costs.controls_machine_rating_cost_coeff
# rotorse.rs.aero_gust.loads_Py
# rotorse.ccblade.Py_b
# rotorse.rp.powercurve.cd_regII
# blade.fatigue.sparU_wohlerA
# rotorse.wt_class.V_extreme50
# blade.internal_structure_2d_fem.d_f
# rotorse.rc.total_cost_utility
# rotorse.rs.y_az
# blade.outer_shape_bem.pitch_axis_yaml
# rotorse.rs.constr.constr_edge_f_margin
# blade.high_level_blade_props.rotor_diameter
# drivese.pitch_I
# blade.opt_var.spar_cap_ps_opt
# costs.generator_mass_cost_coeff
# tcc.rotor_cost
# rotorse.rhoA
# rotorse.rp.cdf.F
# rotorse.ccblade.cl
# rotorse.rs.constr.constr_max_strainL_te
# materials.unit_cost
# costs.converter_mass_cost_coeff
# blade.internal_structure_2d_fem.layer_offset_y_pa_yaml
# rotorse.ccblade.CP
# inn_af.s_opt_L_D
# rotorse.rc.mat_cost
# rotorse.rs.frame.dz
# rotorse.rs.frame.dy
# rotorse.rs.aero_hub_loads.CMb
# rotorse.rc.blade_fixed_cost
# costs.crane_cost
# rotorse.re.te_ss_mats
# costs.fixed_charge_rate
# rotorse.rp.powercurve.rated_efficiency
# blade.opt_var.s_opt_te_ss
# rotorse.ccblade.CM
# blade.high_level_blade_props.presweepTip
# bos.site_assessment_cost
# blade.interp_airfoils.coord_xy_interp
# tcc.tower_parts_cost
# monopile.transition_piece_cost
# rotorse.ccblade.Pz_b
# rotorse.rp.powercurve.V_spline
# rotorse.rp.powercurve.Omega
# blade.high_level_blade_props.presweep
# costs.offset_tcc_per_kW
# rotorse.rs.aero_hub_loads.Mhub
# rotorse.re.precomp.flap_iner
# rotorse.rs.frame.freqs
# rotorse.rs.frame.alpha
# tcc.tower_cost