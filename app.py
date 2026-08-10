import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import io

# ====================================================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ====================================================================
st.set_page_config(
    page_title="Gemelo Digital ACOR 2026 | Termodinámica Pura",
    page_icon="🏭",
    layout="wide"
)

# ====================================================================
# SISTEMA DE SEGURIDAD Y LOGIN
# ====================================================================
def check_password():
    """Devuelve True si el usuario ha introducido la contraseña correcta."""
    def password_entered():
        if st.session_state["password"] == "ACOR2026": 
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h1 style='text-align: center;'>🏭 Acceso Restringido - ACOR 2026</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Por favor, introduce la credencial de ingeniería para acceder al Gemelo Digital.</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.text_input("🔑 Contraseña:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("<h1 style='text-align: center;'>🏭 Acceso Restringido - ACOR 2026</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.text_input("🔑 Contraseña:", type="password", on_change=password_entered, key="password")
            st.error("❌ Contraseña incorrecta. Acceso denegado.")
        return False
    else:
        return True

if not check_password():
    st.stop() 

# ====================================================================
# CATÁLOGOS MAESTROS Y PROPIEDADES TERMODINÁMICAS
# ====================================================================
CATALOGO_VAPORES = {
    'Vapor_Escape': {'entalpia': 502.2},
    'Vapor_1erEfecto': {'entalpia': 521.5},
    'Vapor_2doEfecto': {'entalpia': 525.0},
    'Vapor_3erEfecto': {'entalpia': 528.0},
    'Vapor_4toEfecto': {'entalpia': 533.0},
    'Vapor_5toEfecto': {'entalpia': 536.0},
    'Vapor_6toEfecto': {'entalpia': 542.0},
    'Vapor_Tachas_57C': {'entalpia': 565.0},
    'Condensados_Evaporación': {'entalpia': 94.0}
}

# ====================================================================
# MOTOR MATEMÁTICO: BALANCES DE MASA Y ENERGÍA
# ====================================================================
class PlantaAzucareraCompleta:
    def __init__(self, config):
        self.config = config
        self.cat_vap = CATALOGO_VAPORES
        self.resultados = {}

    def mod_1_difusiones(self):
        c = self.config
        out = {}

        molienda = float(c['IN_Molienda_th'])
        flujo_jugo = molienda * float(c['OP_DifPren_Ratio_Extraccion'])

        fibra_seca_th = molienda * (float(c['IN_Marc_Fibra_pct']) / 100.0)
        pulpa_prensada_th = fibra_seca_th / (float(c['OP_DifPren_MS_PulpaPrensada_pct']) / 100.0)

        f_agua_pren = molienda * (float(c['OP_DifPren_Ratio_AguaPrensas_pct']) / 100.0)
        f_recirc = molienda * (float(c['OP_DifPren_Ratio_Recirculacion_pct']) / 100.0)
        f_desesp = molienda * (float(c['OP_DifPren_Ratio_Desespumador_pct']) / 100.0)

        dt_17 = max(0.0, float(c['OP_DifPren_Int17_TempOut_C']) - float(c['OP_DifPren_Int17_TempIn_C']))
        vap_17 = (f_agua_pren * 1.0 * dt_17) / self.cat_vap['Vapor_5toEfecto']['entalpia']

        dt_18_19 = max(0.0, float(c['OP_DifPren_Int18_19_TempOut_C']) - float(c['OP_DifPren_Int18_19_TempIn_C']))
        vap_18_19 = (f_recirc * 1.0 * dt_18_19) / self.cat_vap['Vapor_5toEfecto']['entalpia']

        dt_20 = max(0.0, float(c['OP_DifPren_Int20_TempOut_C']) - float(c['OP_DifPren_Int20_TempIn_C']))
        vap_20 = (f_desesp * 1.0 * dt_20) / self.cat_vap['Vapor_5toEfecto']['entalpia']

        out.update({
            'OUT_RemolachaProcesada_th': molienda,
            'OUT_Caudal_Jugoverde_Flujo_th': flujo_jugo,
            'OUT_Caudal_Jugoverde_Brix_pct': 17.16,
            'OUT_Caudal_Jugoverde_Pureza_pct': float(c['IN_Pureza_Agricola_pct']),
            'OUT_DifPren_PulpaPrensada_Secado_th': pulpa_prensada_th,
            'OUT_Recalentador17_Vapor_th': vap_17,
            'OUT_Recalentador18_19_Vapor_th': vap_18_19,
            'OUT_Recalentador20_Vapor_th': vap_20
        })
        return out

    def mod_2_calentamiento_verde(self, m1):
        c = self.config
        out = {}

        flujo_jugo = float(m1['OUT_Caudal_Jugoverde_Flujo_th'])
        ds_jugo = float(m1['OUT_Caudal_Jugoverde_Brix_pct'])
        pur_jugo = float(m1['OUT_Caudal_Jugoverde_Pureza_pct'])
        t_in = float(c['OP_DifPren_Temp_Jugoverde_C'])

        cp_jugo = 1.0 - (0.005 * ds_jugo)

        rutas_vapor = {
            '00': 'Vapor_Tachas_57C', 
            '0': 'Vapor_Tachas_57C', 
            '1': 'Vapor_Tachas_57C', 
            '2': 'Vapor_Tachas_57C', 
            '3': 'Vapor_6toEfecto'
        }

        for eq in ['00', '0', '1', '2', '3']:
            t_out = float(c[f'OP_Calverde_Int{eq}_TempOut_C'])
            dt = max(0.0, t_out - t_in)
            fuente = rutas_vapor[eq]
            out[f'OUT_Recalentador{eq}_Vapor_th'] = (flujo_jugo * cp_jugo * dt) / self.cat_vap[fuente]['entalpia']
            t_in = t_out

        out['OUT_Caudal_JugoverdeCaliente_Flujo_th'] = flujo_jugo
        out['OUT_Caudal_JugoverdeCaliente_Brix_pct'] = ds_jugo
        out['OUT_Caudal_JugoverdeCaliente_Pureza_pct'] = pur_jugo
        return out

    def mod_3_depuracion(self, m1, m2, m8):
        c = self.config
        out = {}
        molienda = float(m1['OUT_RemolachaProcesada_th'])
        f_escala = molienda / 445.0
        
        flujo_jugo_entrada = float(m2['OUT_Caudal_JugoverdeCaliente_Flujo_th'])
        brix_entrada = float(m2['OUT_Caudal_JugoverdeCaliente_Brix_pct'])
        pureza_entrada = float(m2['OUT_Caudal_JugoverdeCaliente_Pureza_pct'])
        
        ms_jugo_verde = flujo_jugo_entrada * (brix_entrada / 100.0)
        pol_jugo_verde = ms_jugo_verde * (pureza_entrada / 100.0)
        cp_jugo = 0.94
        
        t_CaO_total = molienda * (float(c['OP_Depuracion_CaO_pct_remolacha']) / 100.0)
        caco3_total = t_CaO_total * (100.0/56.0)
        co2_total = t_CaO_total * (44.0/56.0)
        
        azucar_corefin = float(c['OP_AzucarCorefin_th'])
        azucar_baja = 0.54 * f_escala
        agua_lavado_filtros = 11.47 * f_escala
        agua_lechada_interna = 33.77 * f_escala 
        
        flujo_etapa_2 = flujo_jugo_entrada + agua_lechada_interna + t_CaO_total + azucar_corefin
        
        t_out_3b = float(c['OP_Calent_3B_TempSalida_C'])
        t_out_4 = float(c['OP_Calent_4_TempSalida_C'])
        t_out_56 = float(c['OP_Calent_56_TempSalida_C'])
        t_out_7 = float(c['OP_Calent_7_TempSalida_C'])
        
        out['OUT_Recalentador4_Vapor_th'] = (flujo_etapa_2 * cp_jugo * max(0.0, t_out_4 - t_out_3b)) / self.cat_vap['Vapor_6toEfecto']['entalpia']
        out['OUT_Recalentador5_6_Vapor_th'] = (flujo_etapa_2 * cp_jugo * max(0.0, t_out_56 - t_out_4)) / self.cat_vap['Vapor_5toEfecto']['entalpia']
        out['OUT_Recalentador7_Vapor_th'] = (flujo_etapa_2 * cp_jugo * max(0.0, t_out_7 - t_out_56)) / self.cat_vap['Vapor_4toEfecto']['entalpia']
        
        pol_lost_mud = molienda * 0.0004
        undefined_losses = 1.50 * f_escala 
        impurezas_removidas = ms_jugo_verde * (1.0 - pureza_entrada/100.0) * 0.05 
        
        ms_Lodos_1ro = caco3_total + impurezas_removidas + pol_lost_mud
        Lodos_1ro_humedos = ms_Lodos_1ro / (float(c['OP_PKF_MS_Lodos_pct']) / 100.0)
        
        t_out_1ra_carb = t_out_7 - float(c['OP_Enfriamiento_1raCarb_C'])
        flujo_claro_aprox = flujo_etapa_2 + co2_total - Lodos_1ro_humedos - agua_lechada_interna
        
        t_out_no8 = float(c['OP_Calent_No8_TempSalida_C'])
        out['OUT_Recalentador8_Vapor_th'] = (flujo_claro_aprox * cp_jugo * max(0.0, t_out_no8 - t_out_1ra_carb)) / self.cat_vap['Vapor_5toEfecto']['entalpia']
        
        evap_agua_1ra = 1.75 * f_escala
        evap_agua_2da = 0.80 * f_escala
        vap_5_1ra_th = 0.29 * f_escala
        
        t_out_1ra_filt = t_out_no8 - float(c['OP_Enfriamiento_1raFiltracion_C'])
        flujo_etapa_4 = flujo_claro_aprox + azucar_baja + agua_lavado_filtros
        
        t_out_no9 = float(c['OP_Calent_No9_TempSalida_C'])
        out['OUT_Recalentador9_Vapor_th'] = (flujo_etapa_4 * cp_jugo * max(0.0, t_out_no9 - t_out_1ra_filt)) / self.cat_vap['Vapor_3erEfecto']['entalpia']
        t_out_2da_carb = t_out_no9 - float(c['OP_Enfriamiento_2daCarb_C'])
        
        flujo_jugo_Anteevaporación_total = (flujo_jugo_entrada + t_CaO_total + co2_total + 
                                azucar_corefin + azucar_baja + agua_lavado_filtros + vap_5_1ra_th - 
                                Lodos_1ro_humedos - evap_agua_1ra - evap_agua_2da - undefined_losses)
        
        ms_Anteevaporación_final = ms_jugo_verde + (azucar_corefin * 0.998) + (azucar_baja * 0.98) - impurezas_removidas - pol_lost_mud - undefined_losses
        pol_Anteevaporación_final = pol_jugo_verde + (azucar_corefin * 0.992) + (azucar_baja * 0.95) - pol_lost_mud - undefined_losses
        
        brix_Anteevaporación = (ms_Anteevaporación_final / flujo_jugo_Anteevaporación_total) * 100.0
        pureza_Anteevaporación = (pol_Anteevaporación_final / ms_Anteevaporación_final) * 100.0
        
        flujo_jugo_Anteevaporación_melting = flujo_jugo_Anteevaporación_total * (float(c['OP_JugoAnteevaporación_DestinoMelting_pct']) / 100.0)
        flujo_jugo_Anteevaporación_mod4 = max(0.1, flujo_jugo_Anteevaporación_total - flujo_jugo_Anteevaporación_melting)
        
        out.update({
            'OUT_Caudal_JugoAnteevaporaciónTotal_Flujo_th': flujo_jugo_Anteevaporación_total,
            'OUT_Caudal_JugoAnteevaporaciónTotal_Brix_pct': brix_Anteevaporación,
            'OUT_Caudal_JugoAnteevaporaciónTotal_Pureza_pct': pureza_Anteevaporación,
            'OUT_JugoAnteevaporación_ParaModulo4_Calentamiento_th': flujo_jugo_Anteevaporación_mod4,
            'OUT_JugoAnteevaporación_Temp_C': t_out_2da_carb,
            'OUT_M3_LechadaCal_th': float(agua_lechada_interna + t_CaO_total),
            'OUT_M3_CO2_Consumido_th': float(co2_total),
            'OUT_M3_Recalentador1raCarb_th': float(Lodos_1ro_humedos),
            'OUT_M3_Recalentador2daCarb_th': float(evap_agua_2da + 1.2 * f_escala),
            'OUT_M3_RecalentadorPKF_th': float(Lodos_1ro_humedos * 0.85),
            'OUT_M3_SweetWater_th': float(agua_lechada_interna)
        })
        return out

    def mod_4_calentamiento_jugo_Anteevaporación(self, m3):
        c = self.config
        out = {}
        cp_jugo = 0.96

        flujo_jugo_Anteevaporación = float(m3.get('OUT_JugoAnteevaporación_ParaModulo4_Calentamiento_th', 516.0))
        brix_Anteevaporación = float(m3.get('OUT_Caudal_JugoAnteevaporaciónTotal_Brix_pct', 18.40))
        pur_Anteevaporación = float(m3.get('OUT_Caudal_JugoAnteevaporaciónTotal_Pureza_pct', 91.60))
        temp_entrada = float(m3.get('OUT_JugoAnteevaporación_Temp_C', 87.3))

        t_10 = float(c['OP_Recalentador10_TempSalida_C'])
        out['OUT_Recalentador10_Vapor_th'] = (flujo_jugo_Anteevaporación * cp_jugo * max(0.0, t_10 - temp_entrada)) / self.cat_vap['Vapor_3erEfecto']['entalpia']

        t_11_12 = float(c['OP_Recalentador11_12_TempSalida_C'])
        v11_12 = (flujo_jugo_Anteevaporación * cp_jugo * max(0.0, t_11_12 - t_10)) / self.cat_vap['Vapor_2doEfecto']['entalpia']
        out['OUT_Recalentador11_Vapor_th'] = v11_12 * 0.4
        out['OUT_Recalentador12_Vapor_th'] = v11_12 * 0.6

        t_13 = float(c['OP_Recalentador13_TempSalida_C'])
        out['OUT_Recalentador13_Vapor_th'] = (flujo_jugo_Anteevaporación * cp_jugo * max(0.0, t_13 - t_11_12)) / self.cat_vap['Vapor_1erEfecto']['entalpia']

        t_14 = float(c['OP_Recalentador14_TempSalida_C'])
        out['OUT_Recalentador14_Vapor_th'] = (flujo_jugo_Anteevaporación * cp_jugo * max(0.0, t_14 - t_13)) / self.cat_vap['Vapor_Escape']['entalpia']

        out['OUT_Caudal_JugoAnteevaporaciónCalentado_Flujo_th'] = flujo_jugo_Anteevaporación
        out['OUT_Caudal_JugoAnteevaporaciónCalentado_Brix_pct'] = brix_Anteevaporación
        out['OUT_Caudal_JugoAnteevaporaciónCalentado_Pureza_pct'] = pur_Anteevaporación
        out['OUT_JugoAnteevaporaciónCalentado_Temp_C'] = t_14

        return out

    def mod_6_Cuarto_de_Azucar(self, m1, m7):
        c = self.config
        out = {}
        molienda = float(m1.get('OUT_RemolachaProcesada_th', float(c['IN_Molienda_th'])))
        f_escala = molienda / 445.0
        brix_a = float(c['OP_Cocimiento_BrixMasaA_pct'])

        flujo_liq = float(m7.get('OUT_Caudal_LicorEstandar_Flujo_th', 172.19 * f_escala))
        brix_liq = float(m7.get('OUT_Caudal_LicorEstandar_Brix_pct', 73.90))
        P_liq = float(m7.get('OUT_Caudal_LicorEstandar_Pureza_pct', 93.5))

        def modelo_pan_centrifuga_A(vars, *args):
            F_masa, F_azucar, F_verde, F_Miel_Rica, Agua = vars
            MS_liq, Pol_liq, b_a = args

            MS_Miel_Rica = F_Miel_Rica * 0.780
            Pol_Miel_Rica = MS_Miel_Rica * 0.874
            MS_masa = MS_liq + MS_Miel_Rica
            Pol_masa = Pol_liq + Pol_Miel_Rica

            eq_masa_flow = F_masa - (MS_masa / (b_a / 100.0))

            eq_masa_total = (F_masa + Agua) - (F_azucar + F_verde + F_Miel_Rica)
            MS_az = F_azucar * 0.999
            MS_ver = F_verde * 0.787
            eq_sol = MS_masa - (MS_az + MS_ver + MS_Miel_Rica)

            Pol_az = MS_az * 0.999
            Pol_ver = MS_ver * 0.837
            eq_pol = Pol_masa - (Pol_az + Pol_ver + Pol_Miel_Rica)

            eq_op = F_Miel_Rica - (0.25 * F_verde)
            return [eq_masa_flow, eq_masa_total, eq_sol, eq_pol, eq_op]

        MS_liq = flujo_liq * (brix_liq / 100.0)
        Pol_liq = MS_liq * (P_liq / 100.0)

        est_a = [153.6 * f_escala, 72.5 * f_escala, 64.1 * f_escala, 16.0 * f_escala, 3.2 * f_escala]
        sol_a = fsolve(modelo_pan_centrifuga_A, est_a, args=(MS_liq, Pol_liq, brix_a))
        
        f_masa_a, f_az_comercial_tot, f_verde_a, f_Miel_Rica_a, agua_lav_a = sol_a
        f_polvo = f_az_comercial_tot * (4.61 / (72.50 + 4.61)) 
        f_az_comercial = f_az_comercial_tot - f_polvo

        F_in_b = f_verde_a * (67.53 / 64.14) if f_verde_a > 0 else 67.53 * f_escala
        F_in_c = F_in_b * (32.11 / 67.53) if F_in_b > 0 else 32.11 * f_escala

        def modelo_tachas_b_c(vars, *args):
            F_az_b, F_miel_b, F_az_c, F_melaza = vars
            in_b, in_c = args
            rend_b = 0.481
            rend_c = 0.387
            evap_b = 5.28 * f_escala
            evap_c = -0.81 * f_escala
            agua_c = 2.59 * f_escala
            eq_az_b = F_az_b - (in_b * rend_b)
            eq_miel_b = F_miel_b - (in_b - (in_b * rend_b) - evap_b)
            eq_az_c = F_az_c - (in_c * rend_c)
            eq_melaza = F_melaza - (in_c + agua_c - (in_c * rend_c) - evap_c)
            return [eq_az_b, eq_miel_b, eq_az_c, eq_melaza]

        est_bc = [32.48 * f_escala, 29.77 * f_escala, 12.0 * f_escala, 23.08 * f_escala]
        sol_bc = fsolve(modelo_tachas_b_c, est_bc, args=(F_in_b, F_in_c))
        f_az_b, f_miel_b, f_az_c, f_melaza_final = sol_bc

        masa_b = F_in_b
        masa_c = F_in_c

        dem = (MS_liq + (f_Miel_Rica_a * 0.780))
        pur_a = (Pol_liq + (f_Miel_Rica_a * 0.874)) / dem * 100 if dem > 0 else 93.5

        out.update({
            'OUT_Caudal_MasaCocidaA_Flujo_th': float(f_masa_a),
            'OUT_Caudal_MasaCocidaA_Brix_pct': float(brix_a),
            'OUT_Caudal_MasaCocidaA_Pureza_pct': float(pur_a),
            'OUT_Caudal_AzucarComercial_Flujo_th': float(f_az_comercial),
            'OUT_M6_AzucarPolvo_th': float(f_polvo),
            'OUT_Caudal_AzucarComercial_Brix_pct': 100.0,
            'OUT_Caudal_AzucarComercial_Pureza_pct': 99.9,
            'OUT_Caudal_AzucarB_Fundicion_Flujo_th': float(f_az_b),
            'OUT_Caudal_AzucarB_Fundicion_Pureza_pct': 98.7,
            'OUT_Caudal_MasaCocidaB_Flujo_th': float(masa_b),
            'OUT_Caudal_MasaCocidaB_Brix_pct': float(c['OP_Cocimiento_BrixMasaB_pct']),
            'OUT_Caudal_MasaCocidaB_Pureza_pct': 86.1,
            'OUT_Caudal_MasaCocidaC_Flujo_th': float(masa_c),
            'OUT_Caudal_MasaCocidaC_Brix_pct': float(c['OP_Cocimiento_BrixMasaC_pct']),
            'OUT_Caudal_MasaCocidaC_Pureza_pct': 73.0,
            'OUT_Caudal_MelazaFinal_Flujo_th': float(f_melaza_final),
            'OUT_Caudal_MelazaFinal_Brix_pct': 79.70,
            'OUT_Caudal_MelazaFinal_Pureza_pct': 57.20,
            'OUT_M6_MielVerdeA_th': float(f_verde_a),
            'OUT_M6_MielRicaA_th': float(f_Miel_Rica_a * 0.25),
            'OUT_M6_MielB_th': float(f_miel_b),
            'OUT_M6_AguaCentrifugas_th': float(agua_lav_a),
            'OUT_M6_RendimientoCristal_pct': float(92.4)
        })
        return out

    def mod_7_refundicion(self, m5, m3, m6):
        c = self.config
        out = {}
        flujo_jarabe_evap_th = float(m5.get('OUT_Caudal_Jarabe_Flujo_th', 0.0))
        brix_jarabe_evap = float(m5.get('OUT_Caudal_Jarabe_Brix_pct', 69.40))
        pur_jarabe_evap = float(m5.get('OUT_Caudal_Jarabe_Pureza_pct', 91.60))

        flujo_jugo_Anteevaporación_th = float(m3.get('OUT_Caudal_JugoAnteevaporaciónTotal_Flujo_th', 0.0)) * (float(c['OP_JugoAnteevaporación_DestinoMelting_pct'])/100.0)
        brix_jugo_Anteevaporación_pct = float(m3.get('OUT_Caudal_JugoAnteevaporaciónTotal_Brix_pct', 18.40))
        pur_jugo_Anteevaporación_pct = float(m3.get('OUT_Caudal_JugoAnteevaporaciónTotal_Pureza_pct', 91.60))

        flujo_azucar_b_th = float(m6.get('OUT_Caudal_AzucarB_Fundicion_Flujo_th', 32.48))
        pur_azucar_b = float(m6.get('OUT_Caudal_AzucarB_Fundicion_Pureza_pct', 98.7))

        flujo_polvo = float(m6.get('OUT_M6_AzucarPolvo_th', 4.61))

        ms_jarabe = flujo_jarabe_evap_th * (brix_jarabe_evap / 100.0)
        pol_jarabe = ms_jarabe * (pur_jarabe_evap / 100.0)
        ms_azucar_b = flujo_azucar_b_th * 0.99
        pol_azucar_b = ms_azucar_b * (pur_azucar_b / 100.0)
        ms_jugo_Anteevaporación = flujo_jugo_Anteevaporación_th * (brix_jugo_Anteevaporación_pct / 100.0)
        pol_jugo_Anteevaporación = ms_jugo_Anteevaporación * (pur_jugo_Anteevaporación_pct / 100.0)
        ms_polvo = flujo_polvo * 1.0
        pol_polvo = ms_polvo * 0.998

        masa_seca_total_th = ms_jarabe + ms_azucar_b + ms_polvo + ms_jugo_Anteevaporación
        pol_total_th = pol_jarabe + pol_azucar_b + pol_polvo + pol_jugo_Anteevaporación
        flujo_total_entrante_th = flujo_jarabe_evap_th + flujo_azucar_b_th + flujo_polvo + flujo_jugo_Anteevaporación_th

        fuente_vapor = str(c.get('OP_Recalentador15_Vapor_Fuente', 'Vapor_4toEfecto'))
        temp_salida_C = 91.4
        temp_entrada_C = 89.6
        calor_sensible_Mcal_h = flujo_total_entrante_th * 0.85 * max(0.0, temp_salida_C - temp_entrada_C)
        vapor_requerido_th = calor_sensible_Mcal_h / self.cat_vap[fuente_vapor]['entalpia']

        brix_liquor_estandar = 73.90
        flujo_liquor_estandar_th = masa_seca_total_th / (brix_liquor_estandar / 100.0) if brix_liquor_estandar > 0 else 0.0
        pureza_liquor_estandar = (pol_total_th / masa_seca_total_th) * 100.0 if masa_seca_total_th > 0 else 93.50

        out.update({
            'OUT_Recalentador15_VaporConsumo_th': vapor_requerido_th,
            'OUT_Caudal_LicorEstandar_Flujo_th': flujo_liquor_estandar_th,
            'OUT_Caudal_LicorEstandar_Brix_pct': brix_liquor_estandar,
            'OUT_Caudal_LicorEstandar_Pureza_pct': pureza_liquor_estandar,
            'OUT_Caudal_LicorEstandar_Temp_C': temp_salida_C,
            'OUT_Caudal_CorreAnteevaporaciónEntrante_th': flujo_total_entrante_th,
            'OUT_M7_Entrada_JarabeEvap_th': float(flujo_jarabe_evap_th),
            'OUT_M7_Entrada_AzucarB_th': float(flujo_azucar_b_th),
            'OUT_M7_Entrada_AzucarPolvo_th': float(flujo_polvo),
            'OUT_M7_Entrada_JugoAnteevaporaciónMelting_th': float(flujo_jugo_Anteevaporación_th)
        })
        return out

    def mod_5_evaporacion(self, m4, m3, m6, m7, m1, m2):
        c = self.config
        out = {}
        flujo_entrada = float(m4.get('OUT_Caudal_JugoAnteevaporaciónCalentado_Flujo_th', 500.0))
        temp_entrada = float(m4.get('OUT_JugoAnteevaporaciónCalentado_Temp_C', 123.8))
        brix_entrada = float(m4.get('OUT_Caudal_JugoAnteevaporaciónCalentado_Brix_pct', 18.40))
        if brix_entrada <= 0 or brix_entrada > 40: brix_entrada = 18.40

        SANGRIA_FIJA = 0.30
        D = [0.0] * 6
        d_v1_h13 = float(m4.get('OUT_Recalentador13_Vapor_th', 0.0))
        d_v1_m7 = float(m7.get('OUT_Recalentador15_VaporConsumo_th', 0.0)) if str(c.get('OP_Recalentador15_Vapor_Fuente')) == 'Vapor_1erEfecto' else 0.0
        D[0] = d_v1_h13 + d_v1_m7 + SANGRIA_FIJA

        d_v2_h11 = float(m4.get('OUT_Recalentador11_Vapor_th', 0.0))
        d_v2_h12 = float(m4.get('OUT_Recalentador12_Vapor_th', 0.0))
        d_v2_m7 = float(m7.get('OUT_Recalentador15_VaporConsumo_th', 0.0)) if str(c.get('OP_Recalentador15_Vapor_Fuente')) == 'Vapor_2doEfecto' else 0.0
        d_v2_sec = float(m6.get('OUT_SecaderoAzucar_Vapor_th', 0.0))
        D[1] = d_v2_h11 + d_v2_h12 + d_v2_m7 + d_v2_sec + 3.00 + SANGRIA_FIJA

        d_v3_h10 = float(m4.get('OUT_Recalentador10_Vapor_th', 0.0))
        d_v3_h9 = float(m3.get('OUT_Recalentador9_Vapor_th', 0.0))
        d_v3_tb = float(m6.get('OUT_Vapor3_Demanda_CristalizacionB_th', 0.0))
        d_v3_m7 = float(m7.get('OUT_Recalentador15_VaporConsumo_th', 0.0)) if str(c.get('OP_Recalentador15_Vapor_Fuente')) == 'Vapor_3erEfecto' else 0.0
        D[2] = d_v3_h10 + d_v3_h9 + d_v3_tb + d_v3_m7 + SANGRIA_FIJA

        d_v4_h7 = float(m3.get('OUT_Recalentador7_Vapor_th', 0.0))
        d_v4_ta = float(m6.get('OUT_Vapor4_Demanda_CristalizacionA_th', 0.0))
        d_v4_tc = float(m6.get('OUT_Vapor4_Demanda_CristalizacionC_th', 0.0))
        d_v4_m7 = float(m7.get('OUT_Recalentador15_VaporConsumo_th', 0.0)) if str(c.get('OP_Recalentador15_Vapor_Fuente')) == 'Vapor_4toEfecto' else 0.0
        D[3] = d_v4_h7 + d_v4_ta + d_v4_tc + d_v4_m7 + SANGRIA_FIJA

        d_v5_h56 = float(m3.get('OUT_Recalentador5_6_Vapor_th', 0.0))
        d_v5_h8 = float(m3.get('OUT_Recalentador8_Vapor_th', 0.0))
        d_v5_h17 = float(m1.get('OUT_Recalentador17_Vapor_th', 0.0))
        d_v5_h18 = float(m1.get('OUT_Recalentador18_19_Vapor_th', 0.0))
        d_v5_h20 = float(m1.get('OUT_Recalentador20_Vapor_th', 0.0))
        d_v5_vaho = 0.29 * (float(c['IN_Molienda_th'])/445.0)
        D[4] = d_v5_h56 + d_v5_h8 + d_v5_h17 + d_v5_h18 + d_v5_h20 + d_v5_vaho + SANGRIA_FIJA

        d_v6_h00 = float(m2.get('OUT_Recalentador00_Vapor_th', 0.0))
        d_v6_h0 = float(m2.get('OUT_Recalentador0_Vapor_th', 0.0))
        d_v6_h1 = float(m2.get('OUT_Recalentador1_Vapor_th', 0.0))
        d_v6_h2 = float(m2.get('OUT_Recalentador2_Vapor_th', 0.0))
        d_v6_h3 = float(m2.get('OUT_Recalentador3_Vapor_th', 0.0))
        D[5] = d_v6_h00 + d_v6_h0 + d_v6_h1 + d_v6_h2 + d_v6_h3 + SANGRIA_FIJA

        factor_escala = flujo_entrada / 502.82
        flash_offsets = [8.42, 12.30, 53.36, 55.70, 19.87, 0.0]
        Sangrias_netas = [(D[i] - (flash_offsets[i] * factor_escala)) for i in range(6)]

        T_jugo = [131.6, 127.3, 122.2, 116.5, 110.8, 99.5]
        T_vapor = [136.8, 131.0, 126.4, 120.2, 112.4, 106.5, 94.0]
        V_vivo_0 = 117.34 * factor_escala

        def cp_jugo(brix): return 4.184 * (1.0 - (0.006 * brix))
        def calor_latente(t): return 2501.0 - (2.36 * t)

        def modelo_evaporacion(vars, *args):
            F_out = vars[0:6]; V_evap = vars[6:12]; Brix_out = vars[12:18]
            F_in_0, B_in_0, V_vivo, T_j, T_v, Sangs = args
            eficiencias = [0.980, 0.901, 0.872, 0.455, 0.057, 0.487]
            ecuaciones = []
            for i in range(6):
                F_in = F_in_0 if i == 0 else F_out[i-1]
                Brix_in = B_in_0 if i == 0 else Brix_out[i-1]
                T_jugo_in = temp_entrada if i == 0 else T_j[i-1]
                V_calefaccion = V_vivo if i == 0 else (V_evap[i-1] - Sangs[i-1])
                ecuaciones.append(F_in - F_out[i] - V_evap[i])
                ecuaciones.append((F_in * Brix_in) - (F_out[i] * Brix_out[i]))
                calor_sens = F_in * cp_jugo(Brix_in) * (T_jugo_in - T_j[i])
                evap_flash = calor_sens / calor_latente(T_v[i+1])
                energia_transf = V_calefaccion * calor_latente(T_v[i]) * eficiencias[i]
                evap_transf = energia_transf / calor_latente(T_v[i+1])
                ecuaciones.append(V_evap[i] - (evap_flash + evap_transf))
            return ecuaciones

        est = [400, 300, 200, 150, 140, 133, 100, 100, 50, 50, 5, 15, 23, 31, 46, 61, 63, 69]
        est_escalada = [v * factor_escala if i < 12 else v for i, v in enumerate(est)]
        solucion = fsolve(modelo_evaporacion, est_escalada, args=(flujo_entrada, brix_entrada, V_vivo_0, T_jugo, T_vapor, Sangrias_netas))

        F_out_calc = solucion[0:6]
        V_evap_calc = solucion[6:12]
        Brix_out_calc = solucion[12:18]

        out.update({
            'OUT_Caudal_Jarabe_Flujo_th': float(F_out_calc[-1]),
            'OUT_Caudal_Jarabe_Brix_pct': float(Brix_out_calc[-1]),
            'OUT_Caudal_Jarabe_Pureza_pct': float(m4.get('OUT_Caudal_JugoAnteevaporaciónCalentado_Pureza_pct', 91.60)),
            'OUT_Evaporacion_AguaTotalEvaporada_th': float(sum(V_evap_calc)),
            'OUT_M5_Demanda_V1_Total_th': float(D[0]),
            'OUT_M5_Demanda_V2_Total_th': float(D[1]),
            'OUT_M5_Demanda_V3_Total_th': float(D[2]),
            'OUT_M5_Demanda_V4_Total_th': float(D[3]),
            'OUT_M5_Demanda_V5_Total_th': float(D[4]),
            'OUT_M5_Demanda_V6_Total_th': float(D[5]),
            'OUT_Condensados_Calderas4056_th': float(V_vivo_0),
            'OUT_Condensado_CascadaFinal_9635_th': float(sum(V_evap_calc)),
            'OUT_VaporCalderas_1erEfecto_th': float(V_vivo_0),
            'OUT_M5_CaudalJugoAnteevaporaciónEntrante_th': float(flujo_entrada),
            'OUT_M5_Vapor1erEfecto_th': float(V_vivo_0),
            'OUT_M5_Vapor2doEfecto_th': float(V_evap_calc[0] - Sangrias_netas[0]),
            'OUT_M5_Vapor3erEfecto_th': float(V_evap_calc[1] - Sangrias_netas[1]),
            'OUT_M5_Vapor4toEfecto_th': float(V_evap_calc[2] - Sangrias_netas[2]),
            'OUT_M5_Vapor5toEfecto_th': float(V_evap_calc[3] - Sangrias_netas[3]),
            'OUT_M5_Vapor6toEfecto_th': float(V_evap_calc[4] - Sangrias_netas[4])
        })
        for i in range(6):
            out[f'OUT_M5_Oferta_Ef{i+1}_TOTAL_Generado_th'] = float(V_evap_calc[i])
            out[f'OUT_M5_Salida_Ef{i+1}_Brix_pct'] = float(Brix_out_calc[i])
        return out

    def mod_8_condensados_agua(self, m5, m6, m4, m3, m1, m2, m7):
        out = {}
        reRecalentador_15_vapor_th = float(m7.get('OUT_Recalentador15_VaporConsumo_th', 0.48))
        cond_cascada_evaporacion = float(m5.get('OUT_Condensado_CascadaFinal_9635_th', 246.50))

        fuentes_9635 = [
            {'nombre': 'Condensado Evaporación Cascado', 'flujo_th': cond_cascada_evaporacion},
            {'nombre': 'ReRecalentador Nº 7 (M3)', 'flujo_th': float(m3.get('OUT_Recalentador7_Vapor_th', 0.0))},
            {'nombre': 'ReRecalentador Nº 10 (M4)', 'flujo_th': float(m4.get('OUT_Recalentador10_Vapor_th', 26.44))},
            {'nombre': 'ReRecalentador Nº 11 (M4)', 'flujo_th': float(m4.get('OUT_Recalentador11_Vapor_th', 1.25))},
            {'nombre': 'ReRecalentador Nº 12 (M4)', 'flujo_th': float(m4.get('OUT_Recalentador12_Vapor_th', 1.92))},
            {'nombre': 'ReRecalentador Nº 13 (M4)', 'flujo_th': float(m4.get('OUT_Recalentador13_Vapor_th', 1.98))},
            {'nombre': 'Recalentador Nº 9 (M3)', 'flujo_th': float(m3.get('OUT_Recalentador9_Vapor_th', 6.32))},
            {'nombre': 'Recalentador Nº 4 (M3)', 'flujo_th': float(m3.get('OUT_Recalentador4_Vapor_th', 16.08))},
            {'nombre': 'Recalentadores Nº 5+6 (M3)', 'flujo_th': float(m3.get('OUT_Recalentador5_6_Vapor_th', 5.32))},
            {'nombre': 'Recalentador Nº 8 (M3)', 'flujo_th': float(m3.get('OUT_Recalentador8_Vapor_th', 0.29))},
            {'nombre': 'Agua Prensa Difusión Nº 17 (M1)', 'flujo_th': float(m1.get('OUT_Recalentador17_Vapor_th', 3.04))},
        ]
        total_9635 = sum(item['flujo_th'] for item in fuentes_9635)
        flash_9635 = total_9635 * 0.045
        neto_liquido_9635 = total_9635 - flash_9635

        scrubber_th = float(m1.get('OUT_RemolachaProcesada_th', 445.0)) * (12.0 / 445.0)
        intercambiador_3b_2080_1 = max(0.0, neto_liquido_9635 - scrubber_th)

        fuentes_9620 = [
            {'nombre': 'Tachas B', 'flujo_th': float(m6.get('OUT_Vapor3_Demanda_CristalizacionB_th', 9.84))},
            {'nombre': 'Tachas A y C', 'flujo_th': float(m6.get('OUT_Vapor4_Demanda_CristalizacionA_th', 33.14))},
            {'nombre': 'ReRecalentador Nº 15', 'flujo_th': reRecalentador_15_vapor_th},
            {'nombre': 'Secadero Azúcar', 'flujo_th': float(m6.get('OUT_SecaderoAzucar_Vapor_th', 1.78))},
            {'nombre': 'Int. 18+19 (M1)', 'flujo_th': float(m1.get('OUT_Recalentador18_19_Vapor_th', 2.35))},
            {'nombre': 'Int. 20 (M1)', 'flujo_th': float(m1.get('OUT_Recalentador20_Vapor_th', 1.83))}
        ]
        total_9620 = sum(item['flujo_th'] for item in fuentes_9620)
        flash_9620 = total_9620 * 0.09
        neto_liquido_9620 = total_9620 - flash_9620

        flujo_total_4605 = neto_liquido_9620 + intercambiador_3b_2080_1

        out.update({
            'OUT_Evaporación_FlujoTotal_th': total_9635,
            'OUT_Evaporación_FlujoNetoLiquido_th': neto_liquido_9635,
            'OUT_Intercambiador3B_2080_1_Calculado_th': intercambiador_3b_2080_1,
            'OUT_Deposito9620_FlujoTotal_th': total_9620,
            'OUT_Deposito9620_FlujoNetoLiquido_th': neto_liquido_9620,
            'OUT_Deposito4056_4605_Flujo_th': flujo_total_4605,
            'OUT_Condensados_Totales_th': total_9635 + total_9620 + float(m5.get('OUT_Condensados_Calderas4056_th', 107.73))
        })
        return out

    def mod_9_energia(self, m1, m4, m5, m6):
        c = self.config
        out = {}
        pulpa = float(m1.get('OUT_DifPren_PulpaPrensada_Secado_th', 0.0))
        ms_pulpa_th = pulpa * (float(c['OP_DifPren_MS_PulpaPrensada_pct']) / 100.0)
        pellet = ms_pulpa_th / ((100.0 - float(c['OP_Pulpa_HumedadPellet_pct'])) / 100.0) if c['OP_Pulpa_HumedadPellet_pct'] < 100.0 else 0.0
        agua_evap_sec = max(0.0, pulpa - pellet)

        rend_termico = float(c['OP_SecaderoPulpa_RendimientoTérmico_pct']) / 100.0
        gas_m3h = (agua_evap_sec * 1000.0 * 1.05) / (float(c['OP_SecaderoPulpa_PCI_Gas_kWh_m3']) * rend_termico) if rend_termico > 0.0 else 0.0

        vapor_calderas = float(m5.get('OUT_VaporCalderas_1erEfecto_th', 0.0)) + float(m4.get('OUT_Recalentador14_Vapor_th', 0.0)) + (0.05 * float(c['IN_Molienda_th']))
        mw_elec = (vapor_calderas * float(c['OP_Turbina_ConsumoEspecifico_kWh_tVapor'])) / 1000.0

        vapor_evap_th = float(m5.get('OUT_VaporCalderas_1erEfecto_th', 0.0))
        molienda = float(c['IN_Molienda_th'])

        out.update({
            'OUT_PelletPulpa_Producido_th': pellet,
            'OUT_SecaderoPulpa_AguaEvaporada_th': agua_evap_sec,
            'OUT_SecaderoPulpa_GasNatural_m3h': gas_m3h,
            'OUT_Caldera_VaporVivoTotal_th': vapor_calderas,
            'OUT_Cogeneracion_PotenciaElectrica_MW': mw_elec,
            'OUT_VaporEvap_th': vapor_evap_th,
            'OUT_KPI_RendimientoAzucar_pct': (float(m6.get('OUT_Caudal_AzucarComercial_Flujo_th', 0.0)) / molienda) * 100.0 if molienda > 0 else 0.0
        })
        return out

    def simular(self):
        m1 = self.mod_1_difusiones()
        m2 = self.mod_2_calentamiento_verde(m1)
        m3, m4, m5, m6, m7, m8 = {}, {}, {}, {}, {}, {}

        for _ in range(6):
            m3 = self.mod_3_depuracion(m1, m2, m8)
            m4 = self.mod_4_calentamiento_jugo_Anteevaporación(m3)
            m7 = self.mod_7_refundicion(m5, m3, m6)
            m6 = self.mod_6_Cuarto_de_Azucar(m1, m7)
            m7 = self.mod_7_refundicion(m5, m3, m6)
            m5 = self.mod_5_evaporacion(m4, m3, m6, m7, m1, m2)
            m8 = self.mod_8_condensados_agua(m5, m6, m4, m3, m1, m2, m7)

        m9 = self.mod_9_energia(m1, m4, m5, m6)

        for modulo in [m1, m2, m3, m4, m5, m6, m7, m8, m9]:
            for k, v in modulo.items():
                if isinstance(v, (int, float, np.floating)):
                    modulo[k] = round(float(v), 2)

        self.resultados = {'M1': m1, 'M2': m2, 'M3': m3, 'M4': m4, 'M5': m5, 'M6': m6, 'M7': m7, 'M8': m8, 'M9': m9}
        return self.resultados

# ====================================================================
# DISEÑO DE LA APLICACIÓN WEB EN STREAMLIT
# ====================================================================
st.title("🏭 Gemelo Digital Planta ACOR 2026")
st.markdown("Ajusta los **parámetros de entrada** en la barra lateral y observa en tiempo real los resultados del balance termodinámico y las métricas ejecutivas.")

# BARRA LATERAL (INPUTS)
st.sidebar.header("⚙️ PARÁMETROS DE ENTRADA")

with st.sidebar.expander("🌱 Materia Prima & Módulo 1 (Difusión)", expanded=False):
    in_molienda = st.slider("Molienda t/h", 300.0, 800.0, 445.0, 5.0)
    in_riqueza = st.slider("Riqueza_Remolacha_pct (%)", 12.0, 22.0, 17.4, 0.1)
    in_pureza = st.slider("Pureza_Agricola_pct (%)", 85.0, 95.0, 90.4, 0.1)
    in_marc = st.slider("Marc_Fibra_pct (%)", 3.0, 7.0, 4.5, 0.1)
    op_ratio_ext = st.slider("Ratio_Extraccion", 1.0, 1.3, 1.11, 0.01)
    op_ms_pulpa = st.slider("MS_PulpaPrensada_pct (%)", 20.0, 35.0, 27.5, 0.5)
    op_temp_verde = st.slider("Temp_Jugoverde_C (°C)", 15.0, 40.0, 26.0, 0.5)
    op_ratio_aporte = st.slider("Ratio_AguaAporte_pct (%)", 15.0, 35.0, 24.93, 0.1)
    op_mezcla_caliente = st.slider("Mezcla_AguaCaliente_pct (%)", 50.0, 100.0, 80.0, 1.0)
    op_ratio_prensas = st.slider("Ratio_AguaPrensas_pct (%)", 20.0, 50.0, 37.04, 0.1)
    op_ratio_recirc = st.slider("Ratio_Recirculacion_pct (%)", 100.0, 200.0, 165.0, 1.0)
    op_ratio_desesp = st.slider("Ratio_Desespumador_pct (%)", 20.0, 70.0, 46.0, 1.0)
    op_int17_tin = st.number_input("Int17_TempIn_C (°C)", value=62.0)
    op_int17_tout = st.number_input("Int17_TempOut_C (°C)", value=72.0)
    op_int18_tin = st.number_input("Int18_19_TempIn_C (°C)", value=71.4)
    op_int18_tout = st.number_input("Int18_19_TempOut_C (°C)", value=73.3)
    op_int20_tin = st.number_input("Int20_TempIn_C (°C)", value=71.4)
    op_int20_tout = st.number_input("Int20_TempOut_C (°C)", value=76.5)

with st.sidebar.expander("🔥 Módulo 2 (Calentamiento verde)", expanded=False):
    op_int00_tout = st.number_input("Calverde_Int00_TempOut_C (°C)", value=47.4)
    op_int0_tout = st.number_input("Calverde_Int0_TempOut_C (°C)", value=48.8)
    op_int1_tout = st.number_input("Calverde_Int1_TempOut_C (°C)", value=49.1)
    op_int2_tout = st.number_input("Calverde_Int2_TempOut_C (°C)", value=49.1)
    op_int3_tout = st.number_input("Calverde_Int3_TempOut_C (°C)", value=53.8)
    op_int3a_tout = st.number_input("Calverde_Int3a_TempOut_C (°C)", value=59.0)

with st.sidebar.expander("🧪 Módulo 3 (Depuración y Carb.)", expanded=False):
    op_cao_pct = st.slider("CaO_pct_remolacha (%)", 0.8, 2.0, 1.28, 0.01)
    op_corefin_th = st.number_input("AzucarCorefin_th (t/h)", value=8.80)
    op_alc1_in = st.number_input("1raCarb_AlcEntrada_gh (g/l)", value=2.50)
    op_alc1_out = st.number_input("1raCarb_AlcSalida (g/l)", value=0.90)
    op_alc2_out = st.number_input("2daCarb_AlcSalida (g/l)", value=0.27)
    op_pkf_ms = st.slider("PKF_MS_Lodos_pct (%)", 50.0, 75.0, 64.9, 0.1)
    op_c3b_tin = st.number_input("Calent_3B_TempEntrada_C (°C)", value=61.4)
    op_c3b_tout = st.number_input("Calent_3B_TempSalida_C (°C)", value=65.5)
    op_c4_tout = st.number_input("Calent_4_TempSalida_C (°C)", value=82.3)
    op_c56_tout = st.number_input("Calent_56_TempSalida_C (°C)", value=87.8)
    op_c7_tout = st.number_input("Calent_7_TempSalida_C (°C)", value=87.8)
    op_enf_1carb = st.number_input("Enfriamiento_1raCarb_C (°C)", value=1.60)
    op_enf_1filt = st.number_input("Enfriamiento_1raFiltracion_C (°C)", value=1.00)
    op_enf_2carb = st.number_input("Enfriamiento_2daCarb_C (°C)", value=4.70)
    op_c8_tout = st.number_input("Calent_No8_TempSalida_C (°C)", value=86.2)
    op_c9_tout = st.number_input("Calent_No9_TempSalida_C (°C)", value=92.0)
    op_melting_pct = st.number_input("JugoAnteevaporación_DestinoMelting_pct (%)", value=0.31)

with st.sidebar.expander("♨️ Módulo 4 (Thin Juice Heating)", expanded=False):
    op_c10_tout = st.number_input("Recalentador10_TempSalida_C (°C)", value=117.3)
    op_c11_tout = st.number_input("Recalentador11_12_TempSalida_C (°C)", value=121.6)
    op_c13_tout = st.number_input("Recalentador13_TempSalida_C (°C)", value=123.8)
    op_c14_tout = st.number_input("Recalentador14_TempSalida_C (°C)", value=123.8)

with st.sidebar.expander("💨 Módulo 5 (Evaporación)", expanded=False):
    op_evap_brix_obj = st.slider("Evaporacion_BrixSalida_objetivo_pct (%)", 60.0, 75.0, 69.4, 0.1)

with st.sidebar.expander("🍬 Módulo 6 (Cocimiento) & 9 (Energía)", expanded=False):
    op_brix_masa_a = st.slider("Cocimiento_BrixMasaA_pct (%)", 85.0, 95.0, 91.0, 0.1)
    op_brix_masa_b = st.slider("Cocimiento_BrixMasaB_pct (%)", 90.0, 98.0, 94.6, 0.1)
    op_brix_masa_c = st.slider("Cocimiento_BrixMasaC_pct (%)", 90.0, 98.0, 95.3, 0.1)
    op_pellet_hum = st.slider("Pulpa_HumedadPellet_pct (%)", 5.0, 15.0, 10.0, 0.1)
    op_gas_pci = st.number_input("SecaderoPulpa_PCI_Gas_kWh_m3", value=10.50)
    op_sec_rend = st.slider("SecaderoPulpa_RendimientoTérmico_pct (%)", 70.0, 95.0, 85.0, 1.0)
    op_turb_cons = 95.0

config_usuario = {
    'IN_Molienda_th': in_molienda, 'IN_Riqueza_Remolacha_pct': in_riqueza, 'IN_Pureza_Agricola_pct': in_pureza,
    'IN_Marc_Fibra_pct': in_marc, 'OP_DifPren_Ratio_Extraccion': op_ratio_ext, 'OP_DifPren_MS_PulpaPrensada_pct': op_ms_pulpa,
    'OP_DifPren_Temp_Jugoverde_C': op_temp_verde, 'OP_DifPren_Ratio_AguaAporte_pct': op_ratio_aporte,
    'OP_DifPren_Mezcla_AguaCaliente_pct': op_mezcla_caliente, 'OP_DifPren_Ratio_AguaPrensas_pct': op_ratio_prensas,
    'OP_DifPren_Ratio_Recirculacion_pct': op_ratio_recirc, 'OP_DifPren_Ratio_Desespumador_pct': op_ratio_desesp,
    'OP_DifPren_Int17_TempIn_C': op_int17_tin, 'OP_DifPren_Int17_TempOut_C': op_int17_tout,
    'OP_DifPren_Int18_19_TempIn_C': op_int18_tin, 'OP_DifPren_Int18_19_TempOut_C': op_int18_tout,
    'OP_DifPren_Int20_TempIn_C': op_int20_tin, 'OP_DifPren_Int20_TempOut_C': op_int20_tout,
    'OP_Calverde_Int00_TempOut_C': op_int00_tout, 'OP_Calverde_Int0_TempOut_C': op_int0_tout, 'OP_Calverde_Int1_TempOut_C': op_int1_tout,
    'OP_Recalentador15_Vapor_Fuente': 'Vapor_4toEfecto', 'OP_AzucarCorefin_th': op_corefin_th,
    'OP_Calverde_Int2_TempOut_C': op_int2_tout, 'OP_Calverde_Int3_TempOut_C': op_int3_tout, 'OP_Calverde_Int3a_TempOut_C': op_int3a_tout,
    'OP_Depuracion_CaO_pct_remolacha': op_cao_pct, 'OP_1raCarb_AlcalinidadEntrada_gh': op_alc1_in,
    'OP_1raCarb_AlcalinidadSalida': op_alc1_out, 'OP_2daCarb_AlcalinidadSalida': op_alc2_out, 'OP_PKF_MS_Lodos_pct': op_pkf_ms,
    'OP_Calent_3B_TempEntrada_C': op_c3b_tin, 'OP_Calent_3B_TempSalida_C': op_c3b_tout, 'OP_Calent_4_TempSalida_C': op_c4_tout,
    'OP_Calent_56_TempSalida_C': op_c56_tout, 'OP_Calent_7_TempSalida_C': op_c7_tout, 'OP_Enfriamiento_1raCarb_C': op_enf_1carb,
    'OP_Enfriamiento_1raFiltracion_C': op_enf_1filt, 'OP_Enfriamiento_2daCarb_C': op_enf_2carb, 'OP_Calent_No8_TempSalida_C': op_c8_tout,
    'OP_Calent_No9_TempSalida_C': op_c9_tout, 'OP_JugoAnteevaporación_DestinoMelting_pct': op_melting_pct,
    'OP_Recalentador10_TempSalida_C': op_c10_tout, 'OP_Recalentador11_12_TempSalida_C': op_c11_tout, 'OP_Recalentador13_TempSalida_C': op_c13_tout, 'OP_Recalentador14_TempSalida_C': op_c14_tout,
    'OP_Evaporacion_BrixSalida_objetivo_pct': op_evap_brix_obj,
    'OP_Cocimiento_BrixMasaA_pct': op_brix_masa_a, 'OP_Cocimiento_BrixMasaB_pct': op_brix_masa_b, 'OP_Cocimiento_BrixMasaC_pct': op_brix_masa_c,
    'OP_Pulpa_HumedadPellet_pct': op_pellet_hum, 'OP_SecaderoPulpa_PCI_Gas_kWh_m3': op_gas_pci, 'OP_SecaderoPulpa_RendimientoTérmico_pct': op_sec_rend,
    'OP_Turbina_ConsumoEspecifico_kWh_tVapor': op_turb_cons
}

planta = PlantaAzucareraCompleta(config_usuario)
resultados = planta.simular()

# ====================================================================
# MÉTRICAS KPI SUPERIORES (TARJETAS DINÁMICAS CON COLORES VERDE/ROJO)
# ====================================================================
st.markdown("### 📈 Indicadores Clave de Rendimiento (KPIs)")

molienda_td = in_molienda * 24.0
vap_evap_th = resultados['M9'].get('OUT_VaporEvap_th', 0.0)
correfino_td = float(config_usuario['OP_AzucarCorefin_th']) * 24.0
az_comercial = resultados['M6'].get('OUT_Caudal_AzucarComercial_Flujo_th', 0.0)
pot_mw = resultados['M9'].get('OUT_Cogeneracion_PotenciaElectrica_MW', 0.0)

def render_kpi_card(label, value, target_text, achieved):
    bg_color = "#d4edda" if achieved else "#f8d7da"
    text_color = "#155724" if achieved else "#721c24"
    border_color = "#c3e6cb" if achieved else "#f5c6cb"

    html_code = f"""
    <div style="background-color: {bg_color}; border: 1px solid {border_color}; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px;">
        <span style="color: {text_color}; font-size: 14px; font-weight: bold;">{label}</span>
        <h2 style="color: {text_color}; margin: 8px 0; font-size: 26px;">{value}</h2>
        <span style="color: {text_color}; font-size: 12px;">{target_text}</span>
    </div>
    """
    return html_code

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    ok_m = molienda_td > 10000.0
    st.markdown(render_kpi_card("🌱 Molienda Total", f"{molienda_td:,.0f} T/día", "Objetivo > 10.000 t/día", ok_m), unsafe_allow_html=True)
with kpi2:
    ok_v = vap_evap_th < 115.0
    st.markdown(render_kpi_card("⚡ Vapor a Evaporación", f"{vap_evap_th:.2f} t/h", "Objetivo < 115 t/h", ok_v), unsafe_allow_html=True)
with kpi3:
    ok_c = correfino_td > 200.0
    st.markdown(render_kpi_card("🍬 Correfino Total", f"{correfino_td:,.2f} T/día", "Objetivo > 200 T/día", ok_c), unsafe_allow_html=True)

kpi4, kpi5, kpi6 = st.columns(3)
with kpi4:
    ok_az = az_comercial > 70.0
    st.markdown(render_kpi_card("📦 Azúcar Comercial", f"{az_comercial:.2f} t/h", "Objetivo > 70 t/h", ok_az), unsafe_allow_html=True)
with kpi5:
    st.markdown(render_kpi_card("🔋 Potencia Eléctrica", f"{pot_mw:.2f} MW", "Cogeneración", True), unsafe_allow_html=True)
with kpi6:
    rend_az = resultados['M9'].get('OUT_KPI_RendimientoAzucar_pct', 0.0)
    ok_r = rend_az > 15.0
    st.markdown(render_kpi_card("📦 Rendimiento Azúcar", f"{rend_az:.2f}%", "Objetivo > 15%", ok_r), unsafe_allow_html=True)

st.markdown("---")

# ====================================================================
# PESTAÑAS DE NAVEGACIÓN POR MÓDULOS 
# ====================================================================
tabs = st.tabs([
    "M1: Difusión",
    "M2: Cal verde",
    "M3: Depuración",
    "M4: Jugo Anteevaporación",
    "M5: Evaporación",
    "M7: Refundición",
    "M6: Cocimiento",
    "M8: Condensados",
    "M9: Energía",
    "📄 REPORTE MAESTRO"
])

def render_modulo_tab(mod_key, titulo):
    st.subheader(f"📊 Salidas del Balance: {titulo}")
    dict_data = {k: v for k, v in resultados[mod_key].items() if not isinstance(v, dict)}
    if dict_data:
        df = pd.DataFrame(list(dict_data.items()), columns=['Variable del Proceso', 'Valor Calculado'])
        st.dataframe(df, use_container_width=True, hide_index=True)
    for k, v in resultados[mod_key].items():
        if isinstance(v, dict):
            st.write(f"**Detalle Extra: {k}**")
            st.json(v)

with tabs[0]: render_modulo_tab('M1', 'Módulo 1: Difusiones y Prensas')
with tabs[1]: render_modulo_tab('M2', 'Módulo 2: Calentamiento de Jugo verde')
with tabs[2]: render_modulo_tab('M3', 'Módulo 3: Depuración y Carbonataciones')
with tabs[3]: render_modulo_tab('M4', 'Módulo 4: Thin Juice Heating')
with tabs[4]: render_modulo_tab('M5', 'Módulo 5: Estación de Evaporación')
with tabs[5]: render_modulo_tab('M7', 'Módulo 7: Refundición / Melter House')  
with tabs[6]: render_modulo_tab('M6', 'Módulo 6: Cocimiento y Cristalización')  
with tabs[7]: render_modulo_tab('M8', 'Módulo 8: Condensados y Agua')
with tabs[8]: render_modulo_tab('M9', 'Módulo 9: Secadero y Energía')

# ====================================================================
# REPORTE MAESTRO VISUAL ESTRUCTURADO EN 4 BLOQUES TÉCNICOS
# ====================================================================
with tabs[9]:
    st.subheader("📄 Reporte Maestro de Ingeniería por Módulos (Auditoría Integral)")
    st.markdown("Panel visual interactivo que clasifica el comportamiento operativo en **Datos de Proceso, Laboratorio, Energía y Otros**, con exportación directa a Excel.")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for m_key, m_data in resultados.items():
            flat_data = {k: v for k, v in m_data.items() if not isinstance(v, dict)}
            df_mod = pd.DataFrame(list(flat_data.items()), columns=['Parámetro del Proceso', 'Valor Calculado'])
            df_mod.to_excel(writer, sheet_name=m_key, index=False)

    excel_data = output.getvalue()
    st.download_button(
        label="📥 Descargar Reporte Completo en Excel (.xlsx)",
        data=excel_data,
        file_name="Gemelo_Digital_ACOR_Reporte_Maestro.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")

    def clasificar_variable_bloque(key_name):
        k_lower = key_name.lower()
        if any(w in k_lower for w in ['brix', 'pureza', 'pol_', 'riq']):
            return 'Laboratorio'
        elif any(w in k_lower for w in ['vapor', 'temp', 'calor', 'condensado', 'energia', 'gas', 'potencia', 'mw']):
            return 'Energía'
        elif any(w in k_lower for w in ['flujo', 'th', 'th_', 'th.', 'produccion', 'molienda', 'pulpa', 'pellet', 'lodos', 'lechada', 'co2', 'miel', 'agua', 'jarabe', 'entrada']):
            return 'Proceso'
        else:
            return 'Otros'

    nombres_modulos = {
        'M1': 'Módulo 1: Difusiones y Prensas',
        'M2': 'Módulo 2: Calentamiento de Jugo verde',
        'M3': 'Módulo 3: Depuración y Carbonataciones (Recalentador, Lechada, CO2, Sweet Water)',
        'M4': 'Módulo 4: Thin Juice Heating',
        'M5': 'Módulo 5: Estación de Evaporación (Caudales por efecto y Vapores)',
        'M7': 'Módulo 7: Refundición / Melter House (Desglose de entradas)',  
        'M6': 'Módulo 6: Cocimiento y Cristalización (Mieles, Masas, Aguas, Rendimiento)',  
        'M8': 'Módulo 8: Circuito de Condensados y Agua',
        'M9': 'Módulo 9: Secadero de Pulpa y Energía'
    }

    for m_key, m_titulo in nombres_modulos.items():
        if m_key not in resultados:
            continue

        with st.expander(f"🔹 {m_titulo}", expanded=False):
            bloques = {'Proceso': {}, 'Laboratorio': {}, 'Energía': {}, 'Otros': {}}

            for k, v in resultados[m_key].items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        cat = clasificar_variable_bloque(sub_k)
                        bloques[cat][f"{k} -> {sub_k}"] = sub_v
                else:
                    cat = clasificar_variable_bloque(k)
                    bloques[cat][k] = v

            col_b1, col_b2 = st.columns(2)

            with col_b1:
                st.markdown("##### 🧪 1. Datos de Proceso (Caudales y Fluidos)")
                if bloques['Proceso']:
                    df_proc = pd.DataFrame(list(bloques['Proceso'].items()), columns=['Parámetro', 'Valor'])
                    st.dataframe(df_proc, use_container_width=True, hide_index=True)
                else:
                    st.info("Sin registros de proceso en este módulo.")

                st.markdown("##### 🔬 2. Datos de Laboratorio (Brix, Pureza, No-Azúcares)")
                if bloques['Laboratorio']:
                    df_lab = pd.DataFrame(list(bloques['Laboratorio'].items()), columns=['Parámetro', 'Valor'])
                    st.dataframe(df_lab, use_container_width=True, hide_index=True)
                else:
                    st.info("Sin registros de laboratorio en este módulo.")

            with col_b2:
                st.markdown("##### 🔥 3. Datos de Energía (Vapores, Condensados, Temperaturas)")
                if bloques['Energía']:
                    df_ene = pd.DataFrame(list(bloques['Energía'].items()), columns=['Parámetro', 'Valor'])
                    st.dataframe(df_ene, use_container_width=True, hide_index=True)
                else:
                    st.info("Sin registros de energía en este módulo.")

                st.markdown("##### 📦 4. Otros Parámetros y Balances Secundarios")
                if bloques['Otros']:
                    df_otr = pd.DataFrame(list(bloques['Otros'].items()), columns=['Parámetro', 'Valor'])
                    st.dataframe(df_otr, use_container_width=True, hide_index=True)
                else:
                    st.info("Sin registros adicionales en este módulo.")
