import glob
import os

import numpy as np

import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots


def load_cases(case_names=None):
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
            
    all_data = []
    for idx, log in enumerate(optimization_logs):
        print(f'Loading {idx+1}/{len(optimization_logs)}: {log}')
        data = {}
        cr = om.CaseReader(log)
        cases = cr.get_cases()
        
        for case in cases:
            for key in case.outputs.keys():
                if key not in data.keys():
                    # print(key)
                    data[key] = []
                data[key].append(case.outputs[key])
                
        for key in data.keys():
            data[key] = np.array(data[key])
            
        all_data.append(data)
        
    return all_data, optimization_logs
    
    
    
    
# rotorse.re.precomp.I_all_blades
# blade.high_level_blade_props.presweepTip
# tcc.converter_cost
# rotorse.ccblade.LiftF
# tcc.pitch_system_cost
# blade.opt_var.chord_opt
# costs.platforms_mass_cost_coeff
# tcc.lss_cost
# costs.transformer_mass_cost_coeff
# tcc.yaw_system_cost
# blade.fatigue.teL_wohlerexp
# blade.internal_structure_2d_fem.joint_cost
# rotorse.rc.total_blade_mat_cost_w_waste
# costs.hub_mass_cost_coeff
# bos.site_assessment_cost
# blade.high_level_blade_props.r_blade
# blade.internal_structure_2d_fem.web_offset_y_pa_yaml
# rotorse.re.precomp.blade_mass
# rotorse.ccblade.local_airfoil_velocities
# rotorse.rc.total_cost_utility
# blade.opt_var.spar_cap_ss_opt
# blade.internal_structure_2d_fem.index_layer_start
# rotorse.ccblade.DragF
# rotorse.rc.total_cost_building
# rotorse.ccblade.P
# tcc.transformer_cost
# blade.fatigue.sparU_wohlerexp
# rotorse.ccblade.cd
# blade.internal_structure_2d_fem.layer_offset_y_pa_yaml
# rotorse.rc.mat_cost
# tcc.brake_cost
# costs.elec_connec_machine_rating_cost_coeff
# blade.interp_airfoils.ac_interp
# blade.interp_airfoils.r_thick_interp
# configuration.n_blades
# rotorse.ccblade.Pz_af
# drivese.spinner_cost
# rotorse.rc.blade_variable_cost
# blade.opt_var.twist_opt
# drivese.hss_cost
# tcc.elec_cost
# rotorse.ccblade.L_n_opt
# blade.internal_structure_2d_fem.web_rotation
# blade.internal_structure_2d_fem.layer_rotation_yaml
# inn_af.c_d_opt
# costs.blade_mass_cost_coeff
# rotorse.ccblade.Px_b
# rotorse.ccblade.cd_n_opt
# tcc.spinner_cost
# blade.fatigue.teU_wohlerexp
# costs.tower_mass_cost_coeff
# blade.pa.chord_param
# rotorse.re.precomp.mass_all_blades
# bos.site_assessment_plan_cost
# blade.internal_structure_2d_fem.layer_side
# tcc.hub_system_cost
# rotorse.ccblade.CM
# towerse.monopile_cost
# blade.internal_structure_2d_fem.joint_position
# blade.internal_structure_2d_fem.layer_start_nd_yaml
# rotorse.ccblade.M
# blade.run_inn_af.cl_interp
# bos.design_install_plan_cost
# blade.outer_shape_bem.s_default
# costs.pitch_system_mass_cost_coeff
# rotorse.re.precomp.blade_moment_of_inertia
# inn_af.s_opt_stall_margin
# blade.fatigue.teU_sigma_ult
# blade.internal_structure_2d_fem.layer_rotation
# blade.opt_var.s_opt_spar_cap_ss
# drivese.hub_mat_cost
# tcc.platforms_cost
# tcc.gearbox_cost
# blade.high_level_blade_props.prebend
# inn_af.L_D_opt
# financese.lcoe
# tcc.blade_cost
# rotorse.rc.total_cost_labor
# blade.opt_var.af_position
# costs.gearbox_mass_cost_coeff
# rotorse.stall_check.no_stall_constraint
# rotorse.rc.total_metallic_parts_cost
# tcc.generator_cost
# towerse.unit_cost
# blade.high_level_blade_props.rotor_radius
# costs.offset_tcc_per_kW
# costs.yaw_mass_cost_coeff
# blade.opt_var.s_opt_spar_cap_ps
# inn_af.s_opt_L_D
# tcc.hvac_cost
# blade.high_level_blade_props.rotor_diameter
# blade.high_level_blade_props.presweep
# blade.pa.max_chord_constr
# blade.fatigue.sparL_sigma_ult
# blade.fatigue.sparU_sigma_ult
# blade.opt_var.te_ss_opt
# blade.outer_shape_bem.pitch_axis_yaml
# tcc.nacelle_cost
# rotorse.rc.mat_cost_scrap
# blade.run_inn_af.aoa_inn
# blade.outer_shape_bem.pitch_axis
# tcons.blade_tip_tower_clearance
# tcc.controls_cost
# costs.converter_mass_cost_coeff
# blade.outer_shape_bem.chord_yaml
# costs.lss_mass_cost_coeff
# blade.internal_structure_2d_fem.layer_end_nd
# blade.high_level_blade_props.blade_ref_axis
# blade.fatigue.sparL_wohlerexp
# costs.spinner_mass_cost_coeff
# blade.interp_airfoils.coord_xy_interp
# blade.high_level_blade_props.prebendTip
# blade.opt_var.s_opt_twist
# rotorse.ccblade.D_n_opt
# costs.fixed_charge_rate
# blade.internal_structure_2d_fem.web_offset_y_pa
# blade.internal_structure_2d_fem.definition_web
# bos.port_cost_per_month
# blade.interp_airfoils.cd_interp
# rotorse.rc.cost_capital
# blade.opt_var.s_opt_te_ps
# blade.internal_structure_2d_fem.layer_thickness
# blade.internal_structure_2d_fem.d_f
# blade.pa.twist_param
# costs.controls_machine_rating_cost_coeff
# blade.run_inn_af.cd_interp
# rotorse.rc.total_blade_cost
# inn_af.r_thick_opt
# blade.ps.layer_thickness_param
# tcc.turbine_cost
# rotorse.ccblade.ap
# rotorse.ccblade.Pz_b
# rotorse.rs.constr.constr_max_strainU_spar
# costs.labor_rate
# airfoils.coord_xy
# blade.opt_var.te_ps_opt
# costs.bearing_mass_cost_coeff
# costs.crane_cost
# blade.high_level_blade_props.blade_length
# blade.fatigue.sparL_wohlerA
# blade.outer_shape_bem.chord
# blade.fatigue.teU_wohlerA
# drivese.pitch_cost
# rotorse.ccblade.Py_b
# tcc.main_bearing_cost
# tcc.bedplate_cost
# blade.fatigue.teL_wohlerA
# blade.internal_structure_2d_fem.layer_width_yaml
# blade.internal_structure_2d_fem.web_end_nd_yaml
# blade.opt_var.spar_cap_ps_opt
# costs.hss_mass_cost_coeff
# tcc.turbine_cost_kW
# blade.opt_var.s_opt_chord
# rotorse.rc.total_maintenance_cost
# tcc.tower_cost
# blade.internal_structure_2d_fem.layer_start_nd
# blade.run_inn_af.coord_xy_interp
# tcc.hub_cost
# costs.painting_rate
# tcons.tip_deflection_ratio
# rotorse.ccblade.CP
# rotorse.rc.total_consumable_cost_w_waste
# costs.bos_per_kW
# rotorse.ccblade.alpha
# blade.internal_structure_2d_fem.sigma_max
# drivese.hub_cost
# blade.internal_structure_2d_fem.definition_layer
# rotorse.ccblade.Px_af
# blade.internal_structure_2d_fem.layer_web
# blade.internal_structure_2d_fem.layer_offset_y_pa
# costs.cover_mass_cost_coeff
# costs.hvac_mass_cost_coeff
# blade.fatigue.teL_sigma_ult
# materials.unit_cost
# blade.internal_structure_2d_fem.index_layer_end
# blade.internal_structure_2d_fem.layer_end_nd_yaml
# blade.outer_shape_bem.ref_axis
# rotorse.rc.total_cost_equipment
# rotorse.ccblade.Q
# rotorse.rc.total_cost_tooling
# costs.turbine_number
# blade.interp_airfoils.cl_interp
# blade.interp_airfoils.cm_interp
# blade.outer_shape_bem.twist
# blade.internal_structure_2d_fem.layer_midpoint_nd
# drivese.bedplate_mat_cost
# blade.internal_structure_2d_fem.web_start_nd
# costs.generator_mass_cost_coeff
# blade.outer_shape_bem.af_position
# blade.outer_shape_bem.s
# tcc.tower_parts_cost
# blade.compute_coord_xy_dim.coord_xy_dim
# blade.internal_structure_2d_fem.joint_mass
# blade.compute_reynolds.Re
# rotorse.ccblade.Py_af
# drivese.lss_cost
# towerse.unit_cost_full
# blade.internal_structure_2d_fem.web_start_nd_yaml
# inn_af.s_opt_c_d
# blade.internal_structure_2d_fem.layer_width
# rotorse.rc.blade_fixed_cost
# blade.opt_var.s_opt_te_ss
# rotorse.ccblade.cl_n_opt
# rotorse.ccblade.cl
# towerse.structural_cost
# inn_af.z
# tcc.rotor_cost
# blade.internal_structure_2d_fem.web_rotation_yaml
# rotorse.ccblade.T
# costs.opex_per_kW
# inn_af.stall_margin_opt
# costs.bedplate_mass_cost_coeff
# tcc.cover_cost
# blade.outer_shape_bem.ref_axis_yaml
# bos.boem_review_cost
# rotorse.rs.constr.constr_max_strainL_spar
# inn_af.s_opt_r_thick
# towerse.cm.cost
# rotorse.rp.AEP
# drivese.spinner_mat_cost
# costs.wake_loss_factor
# monopile.transition_piece_cost
# blade.outer_shape_bem.twist_yaml
# drivese.hub_system_cost
# rotorse.ccblade.a
# bos.construction_operations_plan_cost
# towerse.tower_cost
# blade.fatigue.sparU_wohlerA
# blade.internal_structure_2d_fem.web_end_nd
# tcc.hss_cost