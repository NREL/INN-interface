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
        for log in optimization_logs:
            for case_name in case_names:
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
                    data[key].append(case.outputs[key])
            
            for key in data.keys():
                data[key] = np.array(data[key])
                
            # Manually compute t/c and add
            coords = data['blade.run_inn_af.coord_xy_interp']
            maxc = np.max(coords[:, :, :, 1], axis=2)
            minc = np.min(coords[:, :, :, 1], axis=2)
            data['t/c'] = maxc - minc
                
            with open(filename, 'wb') as f:
                dill.dump(data, f)
            
        all_data.append(data)
        
    return all_data, optimization_logs
    
    
if __name__ == "__main__":
    load_cases(just_print_names=False)
    
    
# rotorse.ccblade.Py_b
# blade.interp_airfoils.cd_interp
# costs.controls_machine_rating_cost_coeff
# blade.high_level_blade_props.presweepTip
# tcc.nacelle_cost
# blade.interp_airfoils.coord_xy_interp
# rotorse.ccblade.CM
# rotorse.rc.total_cost_utility
# rotorse.rp.powercurve.rated_pitch
# financese.lcoe
# blade.internal_structure_2d_fem.index_layer_start
# blade.opt_var.af_position
# bos.construction_operations_plan_cost
# blade.fatigue.sparL_wohlerexp
# tcc.platforms_cost
# rotorse.ccblade.Pz_b
# blade.outer_shape_bem.pitch_axis
# tcc.main_bearing_cost
# tcc.rotor_cost
# costs.spinner_mass_cost_coeff
# blade.high_level_blade_props.blade_ref_axis
# blade.internal_structure_2d_fem.layer_width_yaml
# rotorse.rp.AEP
# blade.internal_structure_2d_fem.d_f
# blade.internal_structure_2d_fem.definition_layer
# blade.high_level_blade_props.prebend
# rotorse.rc.total_cost_tooling
# rotorse.rc.total_blade_mat_cost_w_waste
# blade.opt_var.s_opt_twist
# rotorse.ccblade.T
# blade.internal_structure_2d_fem.layer_thickness
# rotorse.ccblade.Py_af
# blade.internal_structure_2d_fem.web_end_nd
# airfoils.coord_xy
# rotorse.ccblade.a
# tcc.spinner_cost
# drivese.pitch_mass
# blade.internal_structure_2d_fem.web_end_nd_yaml
# blade.opt_var.te_ss_opt
# rotorse.rp.powercurve.Cp
# blade.fatigue.teL_wohlerA
# blade.internal_structure_2d_fem.layer_start_nd_yaml
# materials.unit_cost
# tcc.generator_cost
# rotorse.rc.total_maintenance_cost
# blade.outer_shape_bem.twist_yaml
# rotorse.rs.tip_pos.tip_deflection
# blade.pa.chord_param
# costs.bos_per_kW
# blade.high_level_blade_props.rotor_radius
# blade.outer_shape_bem.af_position
# drivese.lss_cost
# costs.elec_connec_machine_rating_cost_coeff
# bos.design_install_plan_cost
# blade.outer_shape_bem.twist
# blade.fatigue.teL_wohlerexp
# tcc.hss_cost
# blade.pa.twist_param
# costs.crane_cost
# tcc.turbine_cost_kW
# rotorse.theta
# blade.fatigue.teU_sigma_ult
# blade.internal_structure_2d_fem.sigma_max
# bos.port_cost_per_month
# inn_af.s_opt_stall_margin
# blade.internal_structure_2d_fem.joint_cost
# blade.internal_structure_2d_fem.web_rotation
# blade.internal_structure_2d_fem.index_layer_end
# rotorse.ccblade.Px_b
# rotorse.rc.total_metallic_parts_cost
# costs.transformer_mass_cost_coeff
# rotorse.re.precomp.I_all_blades
# tcc.elec_cost
# blade.internal_structure_2d_fem.layer_end_nd_yaml
# costs.yaw_mass_cost_coeff
# towerse.structural_cost
# rotorse.rp.powercurve.ax_induct_regII
# blade.fatigue.sparL_wohlerA
# rotorse.ccblade.cd
# costs.blade_mass_cost_coeff
# costs.hvac_mass_cost_coeff
# drivese.spinner_cost
# rotorse.ccblade.Pz_af
# rotorse.rp.powercurve.aoa_regII
# rotorse.rc.cost_capital
# inn_af.L_D_opt
# blade.internal_structure_2d_fem.web_start_nd_yaml
# bos.site_assessment_plan_cost
# drivese.hub_system_cost
# blade.fatigue.sparL_sigma_ult
# rotorse.rc.total_blade_cost
# blade.fatigue.teL_sigma_ult
# blade.high_level_blade_props.rotor_diameter
# costs.tower_mass_cost_coeff
# blade.internal_structure_2d_fem.layer_end_nd
# tcc.yaw_system_cost
# rotorse.rp.powercurve.Cp_aero
# rotorse.rp.powercurve.tang_induct_regII
# blade.opt_var.spar_cap_ss_opt
# inn_af.s_opt_c_d
# tcc.hvac_cost
# blade.internal_structure_2d_fem.layer_start_nd
# blade.outer_shape_bem.ref_axis
# rotorse.ccblade.CP
# tcc.lss_cost
# rotorse.re.precomp.blade_mass
# rotorse.rp.powercurve.L_D
# blade.internal_structure_2d_fem.layer_width
# towerse.cm.cost
# blade.opt_var.s_opt_te_ps
# drivese.hub_cost
# blade.compute_coord_xy_dim.coord_xy_dim
# tcc.brake_cost
# costs.opex_per_kW
# blade.internal_structure_2d_fem.layer_rotation
# hub.pitch_system_scaling_factor
# blade.internal_structure_2d_fem.layer_web
# towerse.tower_cost
# inn_af.r_thick_opt
# rotorse.ccblade.cd_n_opt
# tcc.bedplate_cost
# inn_af.s_opt_r_thick
# rotorse.rs.constr.constr_max_strainU_spar
# tower.layer_thickness
# rotorse.re.precomp.mass_all_blades
# blade.internal_structure_2d_fem.web_offset_y_pa_yaml
# blade.run_inn_af.cl_interp
# inn_af.c_d_opt
# blade.high_level_blade_props.prebendTip
# rotorse.ccblade.ap
# rotorse.rc.total_cost_equipment
# rotorse.ccblade.DragF
# rotorse.rc.blade_fixed_cost
# towerse.monopile_cost
# blade.internal_structure_2d_fem.joint_mass
# rotorse.rp.powercurve.pitch
# blade.opt_var.s_opt_spar_cap_ps
# rotorse.rc.mat_cost
# blade.interp_airfoils.cl_interp
# blade.outer_shape_bem.s_default
# rotorse.rc.blade_variable_cost
# towerse.unit_cost
# costs.converter_mass_cost_coeff
# blade.opt_var.s_opt_te_ss
# inn_af.stall_margin_opt
# blade.ps.layer_thickness_param
# tcc.pitch_system_cost
# rotorse.rp.powercurve.Ct_aero
# costs.cover_mass_cost_coeff
# tcc.hub_system_cost
# drivese.hss_cost
# towerse.unit_cost_full
# drivese.hub_mat_cost
# control.rated_pitch
# blade.outer_shape_bem.chord
# control.max_pitch_rate
# drivese.pitch_cost
# blade.internal_structure_2d_fem.layer_side
# inn_af.z
# rotorse.ccblade.Q
# rotorse.rc.total_cost_labor
# costs.wake_loss_factor
# bos.site_assessment_cost
# tcc.transformer_cost
# tcc.controls_cost
# blade.compute_reynolds.Re
# blade.fatigue.teU_wohlerA
# drivese.bedplate_mat_cost
# tcons.blade_tip_tower_clearance
# blade.internal_structure_2d_fem.joint_position
# blade.outer_shape_bem.pitch_axis_yaml
# costs.hss_mass_cost_coeff
# configuration.n_blades
# costs.painting_rate
# costs.generator_mass_cost_coeff
# costs.gearbox_mass_cost_coeff
# blade.internal_structure_2d_fem.web_rotation_yaml
# blade.interp_airfoils.cm_interp
# tcc.converter_cost
# rotorse.stall_check.no_stall_constraint
# costs.labor_rate
# rotorse.rc.mat_cost_scrap
# blade.fatigue.sparU_wohlerexp
# drivese.spinner_mat_cost
# blade.internal_structure_2d_fem.web_start_nd
# blade.opt_var.te_ps_opt
# blade.outer_shape_bem.s
# costs.turbine_number
# rotorse.ccblade.M
# blade.high_level_blade_props.presweep
# blade.fatigue.teU_wohlerexp
# blade.opt_var.twist_opt
# rotorse.ccblade.P
# rotorse.ccblade.alpha
# rotorse.ccblade.D_n_opt
# tcc.tower_cost
# blade.fatigue.sparU_wohlerA
# blade.high_level_blade_props.blade_length
# monopile.layer_thickness
# rotorse.ccblade.Px_af
# tcc.cover_cost
# blade.outer_shape_bem.chord_yaml
# rotorse.rc.total_consumable_cost_w_waste
# rotorse.rc.total_cost_building
# tcc.turbine_cost
# costs.lss_mass_cost_coeff
# blade.opt_var.s_opt_chord
# costs.offset_tcc_per_kW
# blade.fatigue.sparU_sigma_ult
# blade.run_inn_af.coord_xy_interp
# rotorse.rp.powercurve.Cp_regII
# blade.internal_structure_2d_fem.web_offset_y_pa
# blade.pa.max_chord_constr
# blade.opt_var.s_opt_spar_cap_ss
# costs.hub_mass_cost_coeff
# monopile.transition_piece_cost
# tcons.tip_deflection_ratio
# tcc.blade_cost
# blade.interp_airfoils.r_thick_interp
# inn_af.s_opt_L_D
# costs.platforms_mass_cost_coeff
# rotorse.ccblade.local_airfoil_velocities
# rotorse.rs.constr.constr_max_strainL_spar
# drivese.pitch_I
# blade.opt_var.chord_opt
# blade.internal_structure_2d_fem.layer_offset_y_pa
# blade.opt_var.spar_cap_ps_opt
# rotorse.ccblade.LiftF
# blade.internal_structure_2d_fem.definition_web
# bos.boem_review_cost
# rotorse.ccblade.cl_n_opt
# blade.run_inn_af.aoa_inn
# tcc.tower_parts_cost
# tcc.hub_cost
# costs.fixed_charge_rate
# blade.internal_structure_2d_fem.layer_rotation_yaml
# blade.high_level_blade_props.r_blade
# rotorse.re.precomp.blade_moment_of_inertia
# rotorse.ccblade.cl
# blade.internal_structure_2d_fem.layer_offset_y_pa_yaml
# costs.bedplate_mass_cost_coeff
# rotorse.ccblade.L_n_opt
# costs.bearing_mass_cost_coeff
# costs.pitch_system_mass_cost_coeff
# blade.outer_shape_bem.ref_axis_yaml
# tcc.gearbox_cost
# blade.interp_airfoils.ac_interp
# blade.internal_structure_2d_fem.layer_midpoint_nd
# blade.run_inn_af.cd_interp
