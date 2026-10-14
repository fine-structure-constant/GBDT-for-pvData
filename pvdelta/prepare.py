import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import (A_SPECIES, B_SPECIES, X_SPECIES, additive_features,
                        clean_text, normalize_composition, parse_composition,
                        parse_formula, smiles_heavy_counts)

EXPECTED_FORMULAS = {'MACl':'CH6ClN','MAI':'CH6IN','MABr':'CH6BrN','FAI':'CH5IN2','FACl':'CH5ClN2',
                    'PEAI':'C8H12IN','NH4Cl':'H4ClN','NH4I':'H4IN','NH4SCN':'CH4N2S',
                    'PbCl2':'Cl2Pb','PbI2':'I2Pb','PbBr2':'Br2Pb','SnF2':'F2Sn',
                    'SnCl2':'Cl2Sn','CsI':'CsI','CsCl':'ClCs','CsBr':'BrCs','KI':'IK',
                    'LiI':'ILi','RbI':'IRb','DMSO':'C2H6OS','NMP':'C5H9NO'}


def stable_id(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]


def architecture(value):
    s = clean_text(value).lower()
    if 'tandem' in s or 'stacked' in s:
        return 'tandem'
    if any(k in s for k in ['dssc','dye-sensitized','pec_solar']):
        return 'other_photovoltaic'
    if 'module' in s:
        return 'module'
    if 'n-i-p' in s:
        return 'n_i_p'
    if 'p-i-n' in s:
        return 'p_i_n'
    if 'inverted_flexible' in s:
        return 'p_i_n'
    return 'unspecified_or_other'


def cas_valid(value):
    s = clean_text(value)
    if not s:
        return None
    if not re.fullmatch(r'\d{2,7}-\d{2}-\d', s):
        return False
    digits = s.replace('-','')
    return sum(int(c)*(i+1) for i,c in enumerate(digits[-2::-1])) % 10 == int(digits[-1])


def save_table(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.10g')


def prepare(config, root):
    out = root/config['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    inputs = [p for p in root.glob(config['input_glob']) if not p.name.startswith('~$')]
    if len(inputs) != 1:
        raise ValueError(f'Expected exactly one source workbook in root, found {len(inputs)}')
    path = inputs[0]
    raw = pd.read_excel(path, sheet_name=config['sheet'])
    required = ['solar_cell_structure','perovskite_composition','additive_abbreviation','cas_number','pubchem_id','smiles','molecular_formula','molecular_weight','pce_before','pce_after','pce_delta_abs','pce_delta_relative_pct']
    if list(raw.columns) != required:
        raise ValueError('Source schema differs: review mapping before processing')
    raw.insert(0, 'source_excel_row', np.arange(2, len(raw)+2))
    raw.insert(0, 'record_id', [f'R{i:05d}' for i in range(1,len(raw)+1)])
    save_table(raw, out/'source_snapshot.csv')
    issues = []
    def issue(row, code, severity, field, detail, action):
        issues.append({'record_id':row['record_id'],'source_excel_row':int(row['source_excel_row']),
                       'issue_code':code,'severity':severity,'field':field,'raw_value':clean_text(row.get(field,'')),
                       'detail':detail,'action':action})
    normalized = []
    components = []
    fragments_all = []
    for _, row in raw.iterrows():
        r = row.to_dict()
        for field in required[:8]:
            r[field+'_normalized'] = clean_text(row[field])
        r['architecture'] = architecture(row['solar_cell_structure'])
        comp = parse_composition(row['perovskite_composition'],config['stoichiometry_tolerance'])
        r.update(comp)
        for site in 'ABX':
            for species, coefficient in json.loads(comp[f'{site}_coefficients_json']).items():
                components.append({'record_id':r['record_id'],'source_excel_row':r['source_excel_row'],'site':site,
                                   'species':species,'coefficient':coefficient,
                                   'site_fraction':comp[f'{site}_fraction_{species}'],
                                   'site_ratio_known':comp[f'{site}_ratio_known'],
                                   'parse_status':comp['composition_parse_status'],
                                   'site_assignment':'nominal_not_measured'})
        af, fragments = additive_features(row['molecular_formula'],row['smiles'],row['molecular_weight'])
        r.update(af)
        for frag in fragments:
            fragments_all.append({'record_id':r['record_id'],'source_excel_row':r['source_excel_row'],**frag})
        abbr = clean_text(row['additive_abbreviation'])
        r['additive_name_canonical'] = abbr.replace(' ', '') or '__missing__'
        r['additive_mixture_flag'] = int(bool(re.search(r'[;•·]|\s\+\s',abbr)))
        # CID/formula mappings contain confirmed mismatches. Group by the
        # experimental name so a wrong CID cannot split MACl across folds.
        r['additive_identity_key'] = 'name:'+r['additive_name_canonical']
        r['delta_recomputed_pp'] = row['pce_after']-row['pce_before']
        r['delta_residual_pp'] = row['pce_delta_abs']-r['delta_recomputed_pp']
        r['relative_recomputed_pct'] = 100*r['delta_recomputed_pp']/row['pce_before'] if row['pce_before']>0 else math.nan
        r['additive_mw_relative_error'] = abs(af['additive_mw_g_mol']-af['additive_formula_mw_g_mol'])/af['additive_formula_mw_g_mol'] if af['additive_formula_mw_g_mol']>0 else math.nan
        exclusions = []
        if r['architecture'] not in {'n_i_p','p_i_n'}:
            issue(r,'DEVICE_SCOPE','review','solar_cell_structure','非明确常规单结 n-i-p/p-i-n；不同任务/面积不可直接混合','保留记录，基线暂不纳入')
            exclusions.append('device_scope')
        if min(row['pce_before'],row['pce_after']) <= 0:
            issue(r,'NONPOSITIVE_PCE','review','pce_before','零值可能为失败器件或缺失编码；相对提升可能无定义','保留，待原文核查；基线暂不纳入')
            exclusions.append('nonpositive_pce')
        if r['architecture'] in {'n_i_p','p_i_n'} and max(row['pce_before'],row['pce_after']) > config['single_junction_review_ceiling_pct']:
            issue(r,'PCE_ABOVE_REVIEW_CEILING','critical','pce_after',f'常规单结记录任一 PCE 超过 {config["single_junction_review_ceiling_pct"]}% 审查阈值；可能错字段/错配对','不缩放不截尾，隔离并回查光照条件与原始数据')
            exclusions.append('pce_above_ceiling')
        if abs(r['delta_residual_pp']) > config['delta_tolerance_pp']:
            issue(r,'DELTA_INCONSISTENT','critical','pce_delta_abs','原目标与 pce_after-pce_before 不一致','保留原目标与重算值，基线暂不纳入')
            exclusions.append('delta_inconsistent')
        rel = row['pce_delta_relative_pct']
        if row['pce_before']>0 and (pd.isna(rel) or abs(rel-r['relative_recomputed_pct'])>0.0001):
            issue(r,'RELATIVE_DELTA_INCONSISTENT','review','pce_delta_relative_pct','相对提升与两个 PCE 重算值不符','只审计，不作为特征')
        if 0 < row['pce_before'] < 0.1:
            issue(r,'NEAR_ZERO_BASELINE','review','pce_before','基准效率接近零，导致相对增量爆炸；也可能存在百分比单位混用','不自动乘100；保留绝对增量并核查单位')
            exclusions.append('near_zero_unit_review')
        status = comp['composition_parse_status']
        if status != 'numeric':
            issue(r,'COMPOSITION_'+status.upper(),'review','perovskite_composition','组成未得到完整数值计量；未知比例保留空值','保存已识别物种与原字符串，严格基线暂不纳入')
            exclusions.append('composition_'+status)
        if comp['composition_topology'] not in {'3D_ABX3','double_perovskite','layered_RP_candidate','layered_DJ_candidate'}:
            issue(r,'HOST_STOICHIOMETRY_OR_FAMILY','review','perovskite_composition','计量偏离支持的 ABX3/RP/DJ/双钙钛矿，或属于其他材料家族','不强制归一化为ABX3，严格基线暂不纳入')
            exclusions.append('host_stoichiometry_or_family')
        if not abbr:
            issue(r,'ADDITIVE_NAME_MISSING','review','additive_abbreviation','添加剂名称缺失','保留缺失，不用字符串nan作为类别')
            exclusions.append('additive_name_missing')
        if r['additive_mixture_flag']:
            issue(r,'ADDITIVE_MIXTURE','review','additive_abbreviation','包含多个添加剂或混合描述，单个CID/分子式不一定代表全部组分','保留整条实验，不拆成多条共享标签的伪独立样本')
        if af['additive_formula_status'] != 'parsed' or af['additive_smiles_missing']:
            issue(r,'ADDITIVE_CHEMISTRY_MISSING_OR_UNSUPPORTED','review','molecular_formula','缺少分子式/SMILES或分子式语法不支持','保留缺失指示；仅训练折内填补数值')
        if r['additive_mw_relative_error']>0.01:
            issue(r,'FORMULA_MASS_MISMATCH','review','molecular_weight','分子式理论质量与提供分子量差异超过1%','保留两者；检查盐/水合物/映射错误')
        counts, _, _ = parse_formula(row['molecular_formula'])
        heavy = smiles_heavy_counts(row['smiles'])
        if counts and heavy and {e:v for e,v in counts.items() if e!='H'} != heavy:
            issue(r,'FORMULA_SMILES_ATOM_MISMATCH','critical','smiles','分子式与SMILES的非氢原子计数不一致（仅词法检查，未验证化学图）','隔离添加剂化学描述符，待核查映射')
            exclusions.append('chemical_mapping_conflict')
            for key in af:
                if key not in {'additive_formula_status','additive_smiles_missing','additive_formula_counts_json'}:
                    r[key] = math.nan
        if abbr in EXPECTED_FORMULAS and counts and counts != parse_formula(EXPECTED_FORMULAS[abbr])[0]:
            issue(r,'ABBREVIATION_FORMULA_MISMATCH','critical','molecular_formula',f'明确缩写 {abbr} 与预期分子式 {EXPECTED_FORMULAS[abbr]} 不符','保留名称；屏蔽该行添加剂化学描述符，待核查')
            exclusions.append('chemical_mapping_conflict')
            for key in af:
                if key.startswith('additive_atoms_') or key.startswith('additive_mw_') or key.startswith('additive_formula_mw') or key in {'additive_heavy_atom_count','additive_formula_charge','additive_fragment_count','additive_cation_fragments','additive_anion_fragments','additive_net_formal_charge','additive_ionic_representation'}:
                    r[key] = math.nan
        cv = cas_valid(row['cas_number'])
        if cv is False:
            issue(r,'CAS_INVALID','review','cas_number','CAS格式或校验位错误','保留原值，不作为数值特征')
        if r['additive_mixture_flag']:
            # A single descriptor record cannot describe a multi-additive blend.
            for key in af:
                if key.startswith('additive_atoms_') or key not in {'additive_formula_status','additive_smiles_missing','additive_formula_counts_json'}:
                    r[key] = math.nan
        r['exclusion_reasons'] = ';'.join(sorted(set(exclusions)))
        r['screened_eligible'] = not any(k in exclusions for k in ['device_scope','nonpositive_pce','pce_above_ceiling','delta_inconsistent','additive_name_missing'])
        r['strict_eligible'] = not exclusions
        normalized.append(r)
    data = pd.DataFrame(normalized)
    name_identity_count=data.loc[data.pubchem_id.notna()].groupby('additive_name_canonical').pubchem_id.nunique()
    ambiguous_names=set(name_identity_count[name_identity_count>1].index)
    for _,row in data.loc[data.additive_name_canonical.isin(ambiguous_names)].iterrows():
        issue(row,'NAME_MULTIPLE_CIDS','review','pubchem_id','同一添加剂名称映射到多个PubChem CID，可能有别名/质子化差异或错配','保留原映射供审查；验证分组以实验名称为准')
    original_cols = [c for c in raw.columns if c not in {'record_id','source_excel_row'}]
    duplicate = raw.duplicated(subset=original_cols,keep='first')
    data['exact_duplicate'] = duplicate.astype(int)
    # Canonical repeated observations: do not use targets to define CV groups.
    data['material_pair_group'] = [stable_id('|'.join([r['architecture'],
        json.dumps({s:json.loads(r[f'{s}_coefficients_json']) for s in 'ABX'},sort_keys=True) if r['composition_parse_status']=='numeric' else r['composition_normalized'],r['additive_identity_key']])) for _,r in data.iterrows()]
    data['additive_group'] = data['additive_identity_key'].map(stable_id)
    data['observation_key'] = [stable_id('|'.join([r['material_pair_group'],format(r['pce_before'],'.10g'),format(r['pce_after'],'.10g')])) for _,r in data.iterrows()]
    canonical_duplicate = data.duplicated('observation_key',keep='first')
    data['canonical_duplicate'] = canonical_duplicate.astype(int)
    for i,row in data.iterrows():
        if duplicate.iloc[i]:
            issue(row,'EXACT_DUPLICATE','review','perovskite_composition','整行完全重复','清洗表保留，训练仅保留首次出现')
        elif canonical_duplicate.iloc[i]:
            issue(row,'CANONICAL_DUPLICATE','review','perovskite_composition','规范化后材料组合及前后PCE完全相同','训练仅保留首次出现；可能同一记录重复收集')
    for idx in data.index[canonical_duplicate]:
        data.loc[idx,['strict_eligible','screened_eligible']] = False
        data.loc[idx,'exclusion_reasons'] += ';canonical_duplicate'
    # Same descriptive recipe can legitimately have different outcomes because
    # concentration, process, lab etc. are absent; flag, never average targets.
    conflict_groups = set(data.groupby('material_pair_group')['pce_delta_abs'].agg(lambda s:s.max()-s.min()).loc[lambda s:s>0.1].index)
    data['same_recipe_different_outcomes'] = data.material_pair_group.isin(conflict_groups).astype(int)
    for _,row in data.loc[data.same_recipe_different_outcomes.eq(1)].iterrows():
        issue(row,'RECIPE_OUTCOME_VARIATION','review','pce_delta_abs','相同已知材料/结构组合出现不同结果；缺少剂量/工艺/论文上下文','保留真实变异，按材料组合分组验证，不能按结果择优删除')
    audit = pd.DataFrame(issues)
    counts = audit.groupby(['issue_code','severity','action'],dropna=False).agg(affected_records=('record_id','nunique')).reset_index().sort_values('affected_records',ascending=False)
    counts['fraction_of_records'] = counts.affected_records/len(data)
    field_profile = pd.DataFrame([{'field':c,'dtype':str(raw[c].dtype),'missing_count':int(raw[c].isna().sum()),'missing_fraction':float(raw[c].isna().mean()),'n_unique':int(raw[c].nunique())} for c in original_cols])
    data['issue_codes'] = data.record_id.map(audit.groupby('record_id').issue_code.agg(lambda x:';'.join(sorted(set(x))))).fillna('')
    for name,frame in [('cleaned_records',data),('host_components_long',pd.DataFrame(components)),('additive_fragments_long',pd.DataFrame(fragments_all)),('issues',audit),('issue_summary',counts),('field_profile',field_profile)]:
        save_table(frame,out/f'{name}.csv')
    # Frequency dictionary includes all raw variants for manual review.
    composition_dict = data.groupby(['perovskite_composition_normalized','composition_normalized','composition_parse_status','composition_topology'],dropna=False).agg(record_count=('record_id','size'),example_excel_row=('source_excel_row','first')).reset_index()
    save_table(composition_dict,out/'composition_dictionary.csv')
    sources = {'source_path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sheet':config['sheet'],'rows':len(data),'columns':len(original_cols),'strict_rows':int(data.strict_eligible.sum()),'screened_rows':int(data.screened_eligible.sum()),'paper_count':'unknown: no DOI/paper identifier','source_row_order':'source is sorted by decreasing pce_delta_abs' if raw.pce_delta_abs.is_monotonic_decreasing else 'not monotonic','unavailable_fields':['doi','paper_id','year','additive_concentration','concentration_basis','addition_location','ETL','HTL','solvent','antisolvent','annealing_temperature','annealing_time','device_area','illumination','PCE_scan_protocol','control_treatment_pair_id']}
    (out/'provenance.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(sources,ensure_ascii=False,indent=2),flush=True)
    print(counts[['issue_code','affected_records']].to_string(index=False),flush=True)
    return data, audit, counts, out
