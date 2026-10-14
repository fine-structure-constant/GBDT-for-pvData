"""Conservative, closed-grammar parsing. Never fit unknown stoichiometry.

Site assignment is a nominal compositional convention, not proof of lattice
occupancy. Raw strings, unparsed text and coefficients remain available.
"""
import json
import math
import re
import unicodedata
from collections import Counter

ELEMENTS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U').split()
A_SMALL = ['MA', 'FA', 'Cs', 'Rb', 'K', 'Na', 'Li', 'NH4', 'GA', 'DMA', 'EA']
A_LARGE = ['PEA', 'BA', 'PMA', 'BDA', 'EDA', 'en', 'AVA', 'NMA', 'FPEA', '4FPEA', '4BrPEA', 'PTA']
A_SPECIES = A_SMALL + A_LARGE
X_SPECIES = ['SCN', 'BF4', 'PF6', 'I', 'Br', 'Cl', 'F', 'O', 'S', 'Se']
B_SPECIES = [s for s in ELEMENTS if s not in A_SMALL and s not in X_SPECIES and s not in ['H', 'C', 'N', 'P', 'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']]
TOKENS = sorted(set(A_SPECIES + B_SPECIES + X_SPECIES), key=lambda s: -len(s))
SPECIES_RE = re.compile('|'.join(re.escape(s) for s in TOKENS))
COEFF_RE = re.compile(r'(?:\d+(?:\.\d+)?(?:[+-][xyz])?|[xyz]|\([^()]+\))')
SUBSCRIPTS = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
MASSES = dict(zip(['H','C','N','O','F','P','S','Cl','Br','I','Li','Na','K','Rb','Cs','Mg','Ca','Sr','Ba','Al','Si','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Y','Zr','Mo','Ag','Cd','In','Sn','Sb','Te','La','Ce','Eu','Gd','Yb','Hf','W','Au','Hg','Pb','Bi'],
[1.008,12.011,14.007,15.999,18.998,30.974,32.06,35.45,79.904,126.90447,6.94,22.98977,39.0983,85.4678,132.90545,24.305,40.078,87.62,137.327,26.98154,28.085,47.867,50.9415,51.9961,54.938,55.845,58.933,58.693,63.546,65.38,69.723,72.63,74.9216,78.971,88.9058,91.224,95.95,107.8682,112.414,114.818,118.710,121.760,127.60,138.9055,140.116,151.964,157.25,173.045,178.49,183.84,196.9666,200.592,207.2,208.9804]))


def clean_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    s = unicodedata.normalize('NFKC', str(value).translate(SUBSCRIPTS)).strip()
    return re.sub(r'\s+', ' ', s).replace('−', '-').replace('–', '-')


def normalize_composition(value):
    s = clean_text(value)
    for old, new in [('CH3NH3','MA'), ('HC(NH2)2','FA'), ('CH(NH2)2','FA'), ('Gua','GA'), ('4-FPEA','4FPEA'), ('4F-PEA','4FPEA'), ('4-BrPEA','4BrPEA'), ('5-AVA','AVA')]:
        s = s.replace(old, new)
    return s


def parse_composition(value, tolerance=0.03):
    s = normalize_composition(value)
    result = {'composition_normalized': s}
    multi = bool(re.search(r'[|;•·]', s))
    groups = {'A': {}, 'B': {}, 'X': {}}
    explicit = {'A': [], 'B': [], 'X': []}
    unknown = []
    pos = 0
    if not multi:
        while pos < len(s):
            if s[pos] in '_ -,:':
                pos += 1
                continue
            m = SPECIES_RE.match(s, pos)
            if not m:
                unknown.append(s[pos])
                pos += 1
                continue
            species = m.group()
            pos = m.end()
            # Coefficients written as MA_x / Br_y remain symbolic.
            if pos + 1 < len(s) and s[pos] == '_' and s[pos+1] in 'xyz':
                pos += 1
            cm = COEFF_RE.match(s, pos)
            raw = cm.group() if cm else ''
            if cm:
                pos = cm.end()
            coeff = float(raw) if re.fullmatch(r'\d+(?:\.\d+)?', raw) else (1.0 if not raw else None)
            site = 'A' if species in A_SPECIES else 'X' if species in X_SPECIES else 'B'
            previous = groups[site].get(species, 0)
            groups[site][species] = None if coeff is None or previous is None else previous + coeff
            explicit[site].append(bool(raw))
    complete = bool(s) and not multi and not unknown and all(groups.values())
    all_numeric = complete
    for site, components in groups.items():
        # MA_FA means an unspecified mixture. Ag_Bi in Cs2AgBiBr6 is
        # an ordinary double-perovskite formula; Sn-Pb is unspecified.
        ambiguous_implicit = len(components) > 1 and not any(explicit[site]) and not (site == 'B' and set(components) == {'Ag','Bi'})
        known = bool(components) and not unknown and not multi and not ambiguous_implicit and all(v is not None for v in components.values())
        total = sum(components.values()) if known else math.nan
        result[f'{site}_species'] = ';'.join(sorted(components))
        result[f'{site}_coefficients_json'] = json.dumps(components, sort_keys=True)
        result[f'{site}_sum'] = total
        result[f'{site}_ratio_known'] = int(known)
        result[f'{site}_n_species'] = len(components) if not multi and not unknown else math.nan
        result[f'{site}_mixing_entropy'] = -sum((v/total)*math.log(v/total) for v in components.values() if v > 0) if known and total > 0 else math.nan
        all_numeric = all_numeric and known and total > 0
        vocab = A_SPECIES if site == 'A' else B_SPECIES if site == 'B' else X_SPECIES
        for name in vocab:
            result[f'{site}_has_{name}'] = int(name in components) if not multi and not unknown else math.nan
            result[f'{site}_fraction_{name}'] = (components.get(name, 0)/total) if known and total > 0 else math.nan
    a, b, x = (result[f'{site}_sum'] for site in 'ABX')
    result['A_to_B'] = a/b if all_numeric else math.nan
    result['X_to_B'] = x/b if all_numeric else math.nan
    topology = 'unresolved'
    if all_numeric:
        large = {k:v for k,v in groups['A'].items() if k in A_LARGE and v > 0}
        halide = set(groups['X']).issubset({'I','Br','Cl','F','SCN'})
        near = lambda u,v: abs(u-v) <= tolerance
        if not halide:
            topology = 'non_halide'
        elif not large and near(a/b,1) and near(x/b,3):
            topology = '3D_ABX3'
        elif not large and set(groups['B']) == {'Ag','Bi'} and near(a/b,1) and near(x/b,3):
            topology = 'double_perovskite'
        elif large and near(sum(large.values()),2) and near(a,b+1) and near(x,3*b+1):
            topology = 'layered_RP_candidate'
        elif large and set(large).issubset({'BDA','EDA','en'}) and near(sum(large.values()),1) and near(a,b) and near(x,3*b+1):
            topology = 'layered_DJ_candidate'
        else:
            topology = 'stoichiometry_review'
        if not large and set(groups['B']) == {'Ag','Bi'} and near(a/b,1) and near(x/b,3):
            topology = 'double_perovskite'
    result['composition_parse_status'] = 'multiphase' if multi else 'unparsed' if unknown or not all(groups.values()) else 'symbolic_or_unspecified' if not all_numeric else 'numeric'
    result['composition_unparsed_text'] = ''.join(unknown)
    result['composition_topology'] = topology
    result['nominal_halide_host'] = int(bool(groups['B']) and bool(set(groups['X']) & {'I','Br','Cl'}) and not bool(set(groups['X']) & {'O','S','Se'})) if not multi and not unknown else 0
    # A limited nominal charge check, only with unambiguous standard charges.
    charges = {'Pb':2,'Sn':2,'Ge':2,'Ag':1,'Bi':3}
    if all_numeric and set(groups['B']).issubset(charges) and set(groups['X']).issubset({'I','Br','Cl','F','SCN'}):
        qa = sum(v*(2 if k in {'BDA','EDA','en'} else 1) for k,v in groups['A'].items())
        result['nominal_charge_residual_per_B'] = (qa + sum(charges[k]*v for k,v in groups['B'].items()) - x)/b
    else:
        result['nominal_charge_residual_per_B'] = math.nan
    return result


def parse_formula(value):
    """Parse flat elemental formulas and a terminal charge, reject polymers/groups."""
    s = clean_text(value).replace(' ', '')
    if not s:
        return {}, 'missing', None
    charge = 0
    cm = re.search(r'([+-])(\d*)$', s)
    if cm:
        charge = (1 if cm.group(1) == '+' else -1) * int(cm.group(2) or 1)
        s = s[:cm.start()]
    parts = list(re.finditer(r'([A-Z][a-z]?)(\d*(?:\.\d+)?)', s))
    if ''.join(m.group() for m in parts) != s or any(m.group(1) not in ELEMENTS for m in parts):
        return {}, 'unsupported', None
    counts = Counter()
    for m in parts:
        counts[m.group(1)] += float(m.group(2) or 1)
    return dict(counts), 'parsed', charge


def smiles_fragments(value):
    """Extract only explicit dot fragments/charges, not inferred dissociation."""
    s = clean_text(value)
    rows = []
    for index, fragment in enumerate(s.split('.') if s else []):
        charge = 0
        for atom in re.findall(r'\[([^\]]+)\]', fragment):
            for m in re.finditer(r'(\+{1,}|-{1,})(\d*)', atom):
                charge += (1 if m.group(1)[0] == '+' else -1) * (int(m.group(2)) if m.group(2) else len(m.group(1)))
        rows.append({'fragment_index':index+1,'fragment_smiles':fragment,'formal_charge':charge,'fragment_role':'cation' if charge>0 else 'anion' if charge<0 else 'neutral'})
    return rows


def smiles_heavy_counts(value):
    """Lexical atom inventory only; not chemical graph/sanitization validation."""
    s = clean_text(value)
    if not s:
        return {}
    counts = Counter()
    for token in re.findall(r'\[[^\]]+\]|Br|Cl|[A-Z][a-z]?|[bcnops]', s):
        if token.startswith('['):
            m = re.match(r'\[(?:\d+)?([A-Z][a-z]?|[bcnops])', token)
            if not m:
                return {}
            element = m.group(1).capitalize()
        else:
            element = token.capitalize()
        if element != 'H':
            counts[element] += 1
    return dict(counts)


def additive_features(formula, smiles, molecular_weight):
    counts, status, charge = parse_formula(formula)
    fragments = smiles_fragments(smiles)
    w = clean_text(molecular_weight)
    m = re.fullmatch(r'([\d.]+)\s*(?:g/mol|g mol-1|g·mol-1)?', w)
    mw = float(m.group(1)) if m else math.nan
    theoretical = sum(v*MASSES[e] for e,v in counts.items()) if counts and set(counts).issubset(MASSES) else math.nan
    r = {'additive_formula_status':status,'additive_formula_charge':charge,'additive_mw_g_mol':mw,'additive_formula_mw_g_mol':theoretical,'additive_formula_counts_json':json.dumps(counts,sort_keys=True),
         'additive_smiles_missing':int(not clean_text(smiles)), 'additive_fragment_count':len(fragments) if fragments else math.nan,
         'additive_cation_fragments':sum(f['formal_charge']>0 for f in fragments) if fragments else math.nan,
         'additive_anion_fragments':sum(f['formal_charge']<0 for f in fragments) if fragments else math.nan,
         'additive_net_formal_charge':sum(f['formal_charge'] for f in fragments) if fragments else math.nan,
         'additive_ionic_representation':int(any(f['formal_charge']!=0 for f in fragments)) if fragments else math.nan,
         'additive_heavy_atom_count':sum(v for e,v in counts.items() if e!='H') if counts else math.nan}
    for e in MASSES:
        r[f'additive_atoms_{e}'] = counts.get(e,0) if counts else math.nan
    return r, fragments
